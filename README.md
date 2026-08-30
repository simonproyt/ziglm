# An experinental zig based llm inference engine
curently cpu only but i might make a cuda backend later also i only been able to test it on linux so it might behave diffrent on diffrent oses

#### Models tested
- smollm2-360m
- gemma 4 and its variants with curently only image multimodal working but sound and video might be added in the future


#### How to compile and run 
1. first you need to clone the repo and get the zig compile for your os of choice but this has been only tested on linux so idk if it will work on anything else
2. you need to build it via 
```sh 
zig build -Doptimize=ReleaseFast -Dcpu=native

```
3. now you can try out models that you downloaded 
```sh 
./zig-out/bin/ziglm run -m <path_to_the_model_folder> -p "<prompt>" --greedy -n <number_of_tokens_to_generate>
```
or see the help command for all the options that are possible
```sh
./zig-out/bin/ziglm --help
```

# performance
## CPU
curently according to my tests closely matches  the $\text{TPS} = \frac{\text{System RAM Bandwidth (GB/s)}}{\text{Model Size (GB)}}$  so i dont think i can improve more.
<details>
<summary>see current cpu architechture</summary>

```mermaid
flowchart TD
    %% CLI / Entry
    subgraph Entry ["1. CLI and Server Entrypoints (cli.zig / server.zig)"]
        CLI_RUN["ziglm run / chat / bench"] --> PARSER["CLI Argument Parser<br/>Options, Paths, Thread Count"]
        HTTP_SRV["HTTP REST Server (:8080)<br/>OpenAI-compatible endpoints"] --> PARSER
    end

    %% Engine Initialization & Memory Mapping
    subgraph Init ["2. Engine Initialization and Model Loader (engine.zig / gguf.zig / model.zig)"]
        PARSER --> GGUF_LOAD["GGUF / SafeTensors File Loader<br/>• mmap weights into memory<br/>• Parse tensor metadata and header"]
        GGUF_LOAD --> KV_ALLOC["KV Cache Allocation<br/>layers, kv_heads, max_seq_len, head_size"]
        GGUF_LOAD --> BUF_ALLOC["Per-Thread Scratchpad Buffers<br/>• Residuals, Q/K/V, Attention scores, MLP states"]
        GGUF_LOAD --> TP_INIT["ThreadPool Initialization<br/>• N worker threads pinned to CPU cores<br/>• Lock-free work distribution"]
    end

    %% Multimodal Input Processing (CPU)
    subgraph Inputs ["3. Multimodal Preprocessing and Tower Encoders"]
        direction TB
        TEXT_IN["Prompt Text"] --> TOKENIZER["BPE / SentencePiece Tokenizer<br/>Token ID Lookup (tokenizer.zig)"]
        
        IMAGE_IN["Image File (PNG/JPG/PPM)"] --> IMG_PROC["Image Rescaling [-1, 1]<br/>16x16 Patch Extraction (image.zig)"]
        IMG_PROC --> VIT["16-Layer Vision Transformer<br/>• Patch Proj + 2D Pos Embedding<br/>• Self-Attention + GeGLU<br/>• RMSNorm + mm.input_projection<br/>• 280 Soft Tokens (vision.zig)"]

        VIDEO_IN["Video File (MP4/MKV)"] --> VID_PROC["Frame Extraction and Sampling (video.zig)"]
        VID_PROC --> VID_VIT["Per-Frame Vision Pooling<br/>• 70 Soft Tokens per Frame<br/>• Wrapped with &lt;|video|&gt; + Timestamps"]

        AUDIO_IN["Audio WAV (16kHz Mono)"] --> AUDIO_STFT["Log-Mel Spectrogram (audio.zig)<br/>• 320-pt Hann Window (20ms)<br/>• 512-pt Real FFT + 128 Mel Bins<br/>• Natural Log ln(max(E, 0.001))"]
        AUDIO_STFT --> AUDIO_USM["12-Layer Audio Conformer (audio.zig)<br/>• 2x Conv2D Subsampling (4x temporal reduction)<br/>• Macaron FFNs (0.5 residual weight)<br/>• Relative Attention + Sinusoidal Embeddings<br/>• Causal 1D Depthwise Separable Conv (k=5)<br/>• a.pre_encode.out + RMSNorm + mm.a.input_projection"]
    end

    %% Prefill & Per-Layer Embeddings
    subgraph Prefill ["4. Prefill Sequence Dispatcher (engine.zig / model.zig)"]
        direction TB
        TOKENIZER --> DISPATCH["Prefill Sequence Dispatcher"]
        VIT --> DISPATCH
        VID_VIT --> DISPATCH
        AUDIO_USM --> DISPATCH

        DISPATCH --> PLE_GATE{"Is Multimodal Token?<br/>custom_embedding != null"}
        PLE_GATE -- "Yes (Vision/Audio/Video)" --> PLE_MM["Multimodal PLE Path<br/>• Context Projection ONLY<br/>• No Token ID Identity Lookup"]
        PLE_GATE -- "No (Text Token)" --> PLE_TXT["Text PLE Path<br/>• (Token Identity + Context Proj) * 1/sqrt(2)"]
    end

    %% Transformer Layer CPU Pipeline
    subgraph Transformer ["5. Transformer Decoder Stack (model.zig)"]
        direction TB
        PLE_MM --> LAYER_LOOP["Iterate Layers 0..N-1"]
        PLE_TXT --> LAYER_LOOP

        LAYER_LOOP --> ATTN_NORM["Input RMSNorm<br/>math.rmsNorm()"]
        
        ATTN_NORM --> QKV_PROJ["Parallel Q / K / V GEMV Projections<br/>• Q = x @ W_q (wq.type: Q4_0 / Q8_0 / BF16)<br/>• K = x @ W_k<br/>• V = x @ W_v"]
        
        QKV_PROJ --> ROPE["RoPE Rotary Positional Embeddings<br/>Apply cos/sin frequency rotations"]
        
        ROPE --> KV_UPDATE["Append K, V to Ring KV Cache at pos"]
        
        KV_UPDATE --> MULTI_ATTN["Multi-Head / GQA Self-Attention<br/>• scores = (Q @ K_cache.T) * scale<br/>• Logit Soft-Capping: 50.0 * tanh(scores / 50.0)<br/>• Numerically Stable Softmax(scores)<br/>• context = scores @ V_cache"]
        
        MULTI_ATTN --> ATTN_OUT["Output Projection GEMV: context @ W_o<br/>Residual Add: x = x + attn_out"]
        
        ATTN_OUT --> FFN_NORM["Post-Attention RMSNorm<br/>math.rmsNorm()"]
        
        FFN_NORM --> MLP_BLOCK["Gated MLP (GeGLU / SwiGLU)<br/>• gate = x @ W_gate<br/>• up = x @ W_up<br/>• act = GELU(gate) * up<br/>• down = act @ W_down<br/>Residual Add: x = x + down"]
        
        MLP_BLOCK --> NEXT_LAYER{"More Layers?"}
        NEXT_LAYER -- "Yes" --> LAYER_LOOP
        NEXT_LAYER -- "No" --> FINAL_NORM["Final RMSNorm Normalization"]
    end

    %% Math & SIMD Kernels
    subgraph Compute ["6. CPU SIMD Math and Dequantization Kernels (math.zig / quant.zig)"]
        direction TB
        SIMD_VEC["SIMD Vector Operations<br/>• AVX-512 (512-bit registers, 16 floats/op)<br/>• AVX2 + FMA (256-bit registers, 8 floats/op)<br/>• ARM NEON (128-bit vectors)"]
        
        QUANT_KERNELS["Quantized Dot-Product Kernels<br/>• Q4_0 x F32 Dot Product (Nibble unpacking + Scale)<br/>• Q4_K x F32 Super-Block Dot Product<br/>• Q8_0 x F32 Fast Int8 Dot Product<br/>• BF16 / F16 to F32 Vector Conversions"]
        
        THREAD_PAR["Multi-Threaded Work-Stealing / Chunking<br/>Rows partitioned across worker threads"]
        
        QKV_PROJ -.-> QUANT_KERNELS
        ATTN_OUT -.-> QUANT_KERNELS
        MLP_BLOCK -.-> QUANT_KERNELS
        QUANT_KERNELS -.-> SIMD_VEC
        QUANT_KERNELS -.-> THREAD_PAR
    end

    %% Logits & Sampling
    subgraph Output ["7. Logits and Autoregressive Generation Loop (sampler.zig / engine.zig)"]
        direction TB
        FINAL_NORM --> LM_HEAD["LM Head Output GEMV: x @ W_head<br/>Produces Raw Logits"]
        
        LM_HEAD --> SAMPLER["Logits Processor and Sampler<br/>• Temperature Scaling<br/>• Greedy argmax (if temp == 0)<br/>• Top-K Filtering<br/>• Top-P Nucleus Filtering<br/>• Min-P Filtering"]
        
        SAMPLER --> NEXT_TOKEN["Sample Next Token ID"]
        
        NEXT_TOKEN --> DETOK["Detokenize Token ID &rarr; UTF-8 String<br/>Stream to Stdout / SSE Response Callback"]
        
        NEXT_TOKEN --> EOS_CHECK{"Token == EOS or<br/>pos >= max_tokens?"}
        EOS_CHECK -- "No (Continue Generation)" --> GEN_STEP["pos += 1<br/>Feed Token ID back into Decoder"]
        GEN_STEP --> LAYER_LOOP
        EOS_CHECK -- "Yes (Finished)" --> STATS["Print Bench Stats<br/>(Prefill tok/s, Gen tok/s, Total ms)"]
    end
```

</details>

## CUDA
the current cuda backend is kinda experimental so i havent done a lot of benchmarking stuff yet so its probably unoptimized and i need to work on it

# Dependencies 
- a zig complier (if you want to compile everything from source)
- imagemagick/convert/ffmpeg for image converion for multimodal models
- nvcc for compiling the cuda kernels


more coming soontm               
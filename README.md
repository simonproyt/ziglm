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
# performance
## CPU
curently according to my tests closely matches  the $\text{TPS} = \frac{\text{System RAM Bandwidth (GB/s)}}{\text{Model Size (GB)}}$  so i dont think i can improve more.


# Dependencies 
- a zig complier (if you want to compile everything from source)
- imagemagick/convert/ffmpeg for image converion for multimodal models


more coming soontm               
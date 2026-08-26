# An experinental zig based llm inference engine
curently cpu only but i might make a cuda backend later 

# Models tested
- smollm2-360m
- gemma 4 E2B-it

# How to compile and run 
1. first you need to clone the repo and get the zig compile for your os of choice but this has been only tested on linux so idk if it will work on anything else
2. you need to build it via 
```sh 
zig build -Doptimize=ReleaseFast

```
3. now you can try out models that you downloaded 
```sh 
./zig-out/bin/ziglm run -m <path_to_the_model_folder> -p "<prompt>" --greedy -n <number_of_tokens_to_generate>
```

more coming soontm
# Rustic llama

A [Rust](https://www.rust-lang.org/) 🦀 based llama2 implementation

![rustic_llama](img/poor_lama.jpeg)

* Writen based on [llama2.c](https://github.com/karpathy/llama2.c.git)
* Created with the purpose of understanding the inference process in LLMs.

## Run
```bash 
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin // default model
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin

$ cargo run --release -- --model_path=./stories15M.bin
```

## Current Performance 
50 tok/s 

```
CPU: Intel(R) Core(TM) i7-1065G7 CPU @ 1.30GHz
MemTotal:       32620848 kB
```

![flamegraph](./flamegraph.svg)
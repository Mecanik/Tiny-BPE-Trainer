# Tiny BPE Trainer

A lightweight, header-only **Byte Pair Encoding (BPE)** trainer implemented in modern C++17/20. 

Train your own tokenizer vocabularies compatible with HuggingFace Transformers or use them with [Modern Text Tokenizer](https://github.com/Mecanik/Modern-Text-Tokenizer) for fast, production-ready tokenization in C++.

[![CI](https://github.com/Mecanik/Tiny-BPE-Trainer/actions/workflows/ci.yaml/badge.svg)](https://github.com/Mecanik/Tiny-BPE-Trainer/actions/workflows/ci.yaml)
[![License: MIT](https://img.shields.io/github/license/Mecanik/Tiny-BPE-Trainer.svg)](https://github.com/Mecanik/Tiny-BPE-Trainer/blob/main/LICENSE)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-17%20%7C%2020-blue)](#)
![Header-Only](https://img.shields.io/badge/Header--only-✔️-green)
![No Dependencies](https://img.shields.io/badge/Dependencies-None-brightgreen)
[![Last Commit](https://img.shields.io/github/last-commit/Mecanik/Tiny-BPE-Trainer)](https://github.com/Mecanik/Tiny-BPE-Trainer/commits/main)
[![Stars](https://img.shields.io/github/stars/Mecanik/Tiny-BPE-Trainer?style=social)](https://github.com/Mecanik/Tiny-BPE-Trainer/stargazers)

## Features

- **Full BPE Algorithm**: Train subword vocabularies from scratch
- **Header-Only**: Single file, zero external dependencies
- **High Performance**: Optimized C++ implementation
- **HuggingFace Compatible**: Outputs `vocab.txt` and `merges.txt` files
- **Multiple Formats**: Supports plain text and JSONL input
- **Configurable**: Lowercase, punctuation splitting, normalization
- **CLI Ready**: Complete command-line interface
- **UTF-8 Safe**: Proper Unicode character handling

## Requirements

- **C++17/20** compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- **No external dependencies** - uses only standard library

## Quick Start

### Include the Header

```cpp
#include "Tiny-BPE-Trainer.hpp"
using namespace MecanikDev;
```

### Build the CLI

```bash
g++ -std=c++17 -O3 -o Tiny-BPE-Trainer Tiny-BPE-Trainer.cpp
# or
clang++ -std=c++17 -O3 -o Tiny-BPE-Trainer Tiny-BPE-Trainer.cpp
```

### Basic Training

```cpp
// Initialize trainer
TinyBPETrainer trainer;
trainer
    .set_lowercase(true)
    .set_split_punctuation(true)
    .set_normalize_whitespace(true);

// Train from text file
if (trainer.train_from_file("corpus.txt", 16000, 2)) {
    // Save HuggingFace-compatible files
    trainer.save_vocab("vocab.txt");
    trainer.save_merges("merges.txt");
    
    // Show statistics
    trainer.print_stats();
}
```

### Test Tokenization

```cpp
// Test the trained tokenizer
auto tokens = trainer.tokenize_test("Hello, world!");
// Result: ["Hello", ",", "world", "!</w>"]
```

## Command Line Interface

### Basic Usage

```bash
# Quick demo
./Tiny-BPE-Trainer --demo

# Train from text file
./Tiny-BPE-Trainer -i corpus.txt -v 16000 -o my_tokenizer

# Train from JSONL dataset
./Tiny-BPE-Trainer -i dataset.jsonl --jsonl -v 32000

# Test tokenization
./Tiny-BPE-Trainer --test "Hello, world! This is a test."
```

### All Options

```bash
Options:
  -i, --input <file>      Input text file or JSONL file
  -o, --output <prefix>   Output file prefix (default: "tokenizer")
  -v, --vocab-size <num>  Vocabulary size (default: 32000)  
  -m, --min-freq <num>    Minimum frequency for merges (default: 2)
  --jsonl                 Input is JSONL format
  --text-field <field>    JSONL text field name (default: "text")
  --no-lowercase          Don't convert to lowercase
  --no-punct-split        Don't split punctuation
  --demo                  Run demo with sample data
  --test <text>           Test tokenization on given text
```

## Training Examples

### Small Dataset (1MB)
```bash
./Tiny-BPE-Trainer -i small_corpus.txt -v 8000 -m 2 -o small_tokenizer
# Expected: ~30 seconds, 8K vocabulary
```

### Medium Dataset (100MB)
```bash
./Tiny-BPE-Trainer -i medium_corpus.txt -v 32000 -m 5 -o medium_tokenizer  
# Expected: ~10 minutes, 32K vocabulary
```

### Large Dataset (1GB+)
```bash
./Tiny-BPE-Trainer -i large_corpus.txt -v 50000 -m 10 -o large_tokenizer
# Expected: ~1-2 hours, 50K vocabulary
```

### JSONL Dataset
```bash
./Tiny-BPE-Trainer -i dataset.jsonl --jsonl --text-field content -v 32000
```

### Plain Text
```
The quick brown fox jumps over the lazy dog.
Machine learning is a subset of artificial intelligence.
Natural language processing enables computers to understand human language.
```

### JSONL Format
```jsonl
{"id": 1, "text": "The quick brown fox jumps over the lazy dog."}
{"id": 2, "text": "Machine learning is a subset of artificial intelligence."}
{"id": 3, "text": "Natural language processing enables computers."}
```

### Downloading Corpus with Python (HuggingFace Datasets)

Want to train on real world text like **IMDB reviews**, **Wikipedia**, or **news articles**?

You can use this Python script to download datasets from [HuggingFace Datasets Hub](https://huggingface.co/datasets), and export them into plain `.txt` or `.jsonl` format that works directly with Tiny BPE Trainer.

Don't forget to install the requirements:

```bash
pip install datasets pandas pyarrow
```

#### Save as Plain Text (corpus.txt)

```python
from datasets import load_dataset

# Load dataset (choose from "imdb", "ag_news", "wikitext", etc.)
dataset = load_dataset("imdb", split="train")

with open("corpus.txt", "w", encoding="utf-8") as f:
    for example in dataset:
        text = example.get("text") or example.get("content")
        f.write(text.replace("\n", " ").strip() + "\n")
```

#### Save as JSONL (corpus.jsonl)

```python
import json
from datasets import load_dataset

# Load dataset (choose from "imdb", "ag_news", "wikitext", etc.)
dataset = load_dataset("imdb", split="train")

with open("corpus.jsonl", "w", encoding="utf-8") as f:
    for i, example in enumerate(dataset):
        f.write(json.dumps({"id": i, "text": example["text"]}) + "\n")
```

#### Train with Tiny BPE Trainer

```bash
# Using plain text
./Tiny-BPE-Trainer -i corpus.txt -v 16000 -m 2 -o imdb_tokenizer

# Using JSONL
./Tiny-BPE-Trainer -i corpus.jsonl --jsonl -v 16000 -o imdb_tokenizer
```

## Output Files

### vocab.txt (HuggingFace Compatible)
```
<|endoftext|>
<|unk|>
<|pad|>  
<|mask|>
!
"
#
...
the
of
and
ing</w>
er</w>
...
```

### merges.txt (BPE Rules)
```
#version: 0.2
i n
t h
th e
e r
...
```

## API Reference

### Core Methods

```cpp
class TinyBPETrainer {
    // Configuration
    TinyBPETrainer& set_lowercase(bool enable);
    TinyBPETrainer& set_split_punctuation(bool enable);  
    TinyBPETrainer& set_normalize_whitespace(bool enable);
    TinyBPETrainer& set_special_tokens(eos, unk, pad, mask);
    
    // Training
    bool train_from_file(filepath, vocab_size=32000, min_freq=2);
    bool train_from_jsonl(filepath, text_field="text", vocab_size=32000, min_freq=2);
    
    // Output
    bool save_vocab(vocab_path);
    bool save_merges(merges_path);
    void print_stats();
    
    // Testing  
    std::vector<std::string> tokenize_test(text);
};
```

### Configuration Options

```cpp
TinyBPETrainer trainer;

trainer
    .set_lowercase(true)              // Convert to lowercase
    .set_split_punctuation(true)      // Split on punctuation  
    .set_normalize_whitespace(true)   // Normalize whitespace
    .set_special_tokens(              // Custom special tokens
        "<|endoftext|>", 
        "<|unk|>", 
        "<|pad|>", 
        "<|mask|>"
    );
```

## Integration with Tokenizers

### Use with Modern Text Tokenizer

```cpp
#include "Modern-Text-Tokenizer.hpp" // Tokenizer
#include "Tiny-BPE-Trainer.hpp"    // BPE trainer

using namespace MecanikDev;

// Train BPE vocabulary
TinyBPETrainer trainer;
trainer.train_from_file("corpus.txt", 16000);
trainer.save_vocab("my_vocab.txt");
trainer.save_merges("my_merges.txt");

// Use with tokenizer 
TextTokenizer tokenizer;
tokenizer.load_vocab("my_vocab.txt");
auto token_ids = tokenizer.encode("Hello, world!");
```

### Use with HuggingFace

```python
# Python - load in HuggingFace Tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE

# Load our trained BPE
tokenizer = Tokenizer(BPE(
    vocab="my_vocab.txt", 
    merges="my_merges.txt"
))

tokens = tokenizer.encode("Hello, world!")
```

## Performance

```bash
Starting BPE training...
   Input: imdb.txt
   Format: Plain text
   Vocab size: 32000
   Min frequency: 2
   Output prefix: tokenizer
Reading corpus from: imdb.txt
Processed 33157823 characters, 6952632 words
Unique word forms: 106008
Initial vocabulary size: 240
Starting BPE training...
    ...
BPE training completed!
   Final vocabulary size: 32000
   Total merges: 31760
   Training time: 1962 seconds
Saved vocabulary (32000 tokens) to: tokenizer_vocab.txt
Saved merges (31760 rules) to: tokenizer_merges.txt

Training completed successfully!
   Total time: 1966 seconds

Training Statistics:
   Characters processed: 33157823
   Words processed: 6952632
   Final vocab size: 32000
   BPE merges: 31760
   Compression ratio: 0.0010
```

*Benchmark on AMD Ryzen 9 5900X, compiled with -O3.*

## Algorithm Details

### BPE Training Process

1. **Preprocessing**
   - Normalize whitespace  
   - Convert to lowercase (optional)
   - Split punctuation (optional)

2. **Character Initialization**
   ```
   "hello" → ["h", "e", "l", "l", "o", "</w>"]
   ```

3. **Iterative Merging**
   ```
   Most frequent pair: "l" + "l" → "ll"
   "hello" → ["h", "e", "ll", "o", "</w>"]
   ```

4. **Vocabulary Building**
   - Characters: `h`, `e`, `l`, `o`, `</w>`
   - Merges: `ll`, `he`, `ell`, `hello`
   - Special tokens: `<|unk|>`, `<|pad|>`, etc.

### Key Features

- **Subword Units**: Handles unknown words through decomposition
- **Frequency-Based**: Most common patterns get merged first  
- **Deterministic**: Same corpus always produces same vocabulary
- **Compression**: Reduces vocabulary size vs. word-level tokenization

## Troubleshooting

### Common Issues

**"Training failed" Error**
```bash
# Check file exists and is readable
ls -la corpus.txt
file corpus.txt

# Try smaller vocabulary size
./Tiny-BPE-Trainer -i corpus.txt -v 8000 -m 1
```

**Slow Training**
```bash
# Increase minimum frequency
./Tiny-BPE-Trainer -i corpus.txt -v 32000 -m 10

# Use smaller corpus for testing
head -n 10000 large_corpus.txt > small_test.txt
```

**Memory Issues**
```bash
# Monitor memory usage
top -p $(pgrep Tiny-BPE-Trainer)

# Reduce vocabulary size
./Tiny-BPE-Trainer -i corpus.txt -v 16000
```

### Performance Tips

1. **Start Small**: Test with small corpus and vocabulary first
2. **Adjust min_frequency**: Higher values = faster training, smaller vocab
3. **Preprocessing**: Clean your corpus for better results
4. **Incremental**: Train smaller models first, then scale up

## Roadmap

### Planned Features

- [ ] **Parallel Training**: Multi-threaded BPE training
- [ ] **Streaming Mode**: Process huge files without loading into memory  
- [ ] **Advanced Preprocessing**: Custom regex patterns, language-specific rules
- [ ] **Evaluation Metrics**: Compression ratio, OOV handling statistics
- [ ] **Visualization**: Plot vocabulary growth, merge frequency distributions
- [ ] **Export Formats**: SentencePiece, custom binary formats

### Future Considerations

- [ ] **Tokenizer Integration**: Seamless loading of trained BPE models
- [ ] **HuggingFace Plugin**: Direct integration with transformers library
- [ ] **TensorFlow/PyTorch**: C++ ops for training integration

## Contributing

We welcome contributions! Areas of interest:

1. **Performance**: SIMD optimizations, better algorithms
2. **Features**: New preprocessing options, export formats
3. **Testing**: More edge cases, different languages
4. **Documentation**: Tutorials, examples, use cases

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Inspired by open-source libraries like [SentencePiece](https://github.com/google/sentencepiece) and [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)
- Format compatibility modeled after HuggingFace's `vocab.txt` and `merges.txt` outputs
- Based on the original [Byte Pair Encoding paper](https://arxiv.org/abs/1508.07909) by Sennrich
- UTF-8 safety and normalization techniques informed by modern C++ text processing resources

## Learn More

- [BPE Paper](https://arxiv.org/abs/1508.07909) - Original Byte Pair Encoding paper
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [SentencePiece](https://github.com/google/sentencepiece) - Google's implementation  
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) - Fast tokenization library

---

**⭐ Star this repo if you find it useful!**

Built with ❤️ for the C++ and NLP community
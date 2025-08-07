#include "Tiny-BPE-Trainer.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>

using namespace std;
using namespace MecanikDev;

void print_banner() {
	std::cout << R"(
 ████████╗██╗███╗   ██╗██╗   ██╗    ██████╗ ██████╗ ███████╗
 ╚══██╔══╝██║████╗  ██║╚██╗ ██╔╝    ██╔══██╗██╔══██╗██╔════╝
    ██║   ██║██╔██╗ ██║ ╚████╔╝     ██████╔╝██████╔╝█████╗  
    ██║   ██║██║╚██╗██║  ╚██╔╝      ██╔══██╗██╔═══╝ ██╔══╝  
    ██║   ██║██║ ╚████║   ██║       ██████╔╝██║     ███████╗
    ╚═╝   ╚═╝╚═╝  ╚═══╝   ╚═╝       ╚═════╝ ╚═╝     ╚══════╝
    
    Tiny BPE Trainer - Modern C++ Implementation
    Header-only • Fast • HuggingFace Compatible
)" << std::endl;
}

void print_help() {
	std::cout << R"(
Usage: Tiny-BPE-Trainer [options]

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
  -h, --help             Show this help

Examples:
  # Train from text file
  ./Tiny-BPE-Trainer -i corpus.txt -v 16000 -o my_tokenizer
  
  # Train from JSONL dataset  
  ./Tiny-BPE-Trainer -i dataset.jsonl --jsonl -v 32000
  
  # Test tokenization
  ./Tiny-BPE-Trainer --test "Hello, world! This is a test."
  
  # Run interactive demo
  ./Tiny-BPE-Trainer --demo

Output files:
  - <prefix>_vocab.txt    Vocabulary file (HuggingFace compatible)
  - <prefix>_merges.txt   BPE merge rules
)" << std::endl;
}

void create_sample_corpus(const std::string& filename) {
	std::ofstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Error: Cannot create sample corpus file" << std::endl;
		return;
	}

	// Sample text corpus for demonstration
	std::vector<std::string> sample_texts = {
		"The quick brown fox jumps over the lazy dog.",
		"Machine learning is a subset of artificial intelligence.",
		"Natural language processing enables computers to understand human language.",
		"Tokenization is the process of breaking text into smaller units called tokens.",
		"Byte Pair Encoding is a data compression technique adapted for tokenization.",
		"Deep learning models require large amounts of training data.",
		"Transformers have revolutionized the field of natural language processing.",
		"BERT, GPT, and T5 are popular transformer-based language models.",
		"Preprocessing text data is crucial for successful machine learning.",
		"Vocabulary size affects both model performance and computational requirements.",
		"Subword tokenization helps handle out-of-vocabulary words effectively.",
		"The attention mechanism allows models to focus on relevant parts of input.",
		"Fine-tuning pre-trained models is more efficient than training from scratch.",
		"Evaluation metrics help measure model performance on specific tasks.",
		"Data augmentation techniques can improve model robustness and generalization."
	};

	// Repeat the texts multiple times to create a larger corpus
	for (int i = 0; i < 100; ++i) {
		for (const auto& text : sample_texts) {
			file << text << "\n";
		}
	}

	std::cout << "Created sample corpus: " << filename << std::endl;
}

void run_demo() {
	std::cout << "\nRunning Tiny BPE Trainer Demo\n" << std::endl;

	// Create sample corpus
	std::string corpus_file = "demo_corpus.txt";
	create_sample_corpus(corpus_file);

	// Initialize trainer with configuration
	TinyBPETrainer trainer;
	trainer
		.set_lowercase(true)
		.set_split_punctuation(true)
		.set_normalize_whitespace(true);

	std::cout << "\nTraining BPE with demo corpus..." << std::endl;

	// Train BPE
	if (trainer.train_from_file(corpus_file, 1000, 2)) {
		// Save outputs
		trainer.save_vocab("demo_vocab.txt");
		trainer.save_merges("demo_merges.txt");

		// Show statistics
		trainer.print_stats();

		// Test tokenization
		std::cout << "\nTesting tokenization:" << std::endl;

		std::vector<std::string> test_sentences = {
			"Hello, world!",
			"Machine learning is fascinating.",
			"The transformer architecture is powerful.",
			"Out-of-vocabulary words are challenging."
		};

		for (const auto& sentence : test_sentences) {
			auto tokens = trainer.tokenize_test(sentence);
			std::cout << "  Input: \"" << sentence << "\"" << std::endl;
			std::cout << "  Tokens: ";
			for (size_t i = 0; i < tokens.size(); ++i) {
				std::cout << "\"" << tokens[i] << "\"";
				if (i < tokens.size() - 1) std::cout << ", ";
			}
			std::cout << " (" << tokens.size() << " tokens)" << std::endl << std::endl;
		}

		std::cout << "Demo completed! Check demo_vocab.txt and demo_merges.txt" << std::endl;
	}
	else {
		std::cout << "Demo training failed!" << std::endl;
	}

	// Cleanup
	std::remove(corpus_file.c_str());
}

void test_tokenization(const std::string& text) {
	std::cout << "\nTesting tokenization (no trained model)" << std::endl;
	std::cout << "Note: This shows basic preprocessing only." << std::endl;
	std::cout << "For full BPE tokenization, train a model first.\n" << std::endl;

	TinyBPETrainer trainer;
	trainer
		.set_lowercase(true)
		.set_split_punctuation(true)
		.set_normalize_whitespace(true);

	// This is a basic demonstration - full tokenization requires a trained model
	std::cout << "Input: \"" << text << "\"" << std::endl;

	// We can show the preprocessing steps
	std::string normalized = text;

	// Simulate normalization
	std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);

	// Basic word splitting
	std::istringstream iss(normalized);
	std::string word;
	std::vector<std::string> words;

	while (iss >> word) {
		words.push_back(word);
	}

	std::cout << "Preprocessed words: ";
	for (size_t i = 0; i < words.size(); ++i) {
		std::cout << "\"" << words[i] << "\"";
		if (i < words.size() - 1) std::cout << ", ";
	}
	std::cout << std::endl;

	std::cout << "\nTo see full BPE tokenization, train a model first:" << std::endl;
	std::cout << "   ./bpe_trainer --demo" << std::endl;
	std::cout << "   ./bpe_trainer -i your_corpus.txt -v 16000" << std::endl;
}

int main(int argc, char* argv[]) {
	print_banner();

	// Parse command line arguments
	std::string input_file;
	std::string output_prefix = "tokenizer";
	int vocab_size = 32000;
	int min_frequency = 2;
	bool is_jsonl = false;
	std::string text_field = "text";
	bool use_lowercase = true;
	bool split_punctuation = true;
	bool demo_mode = false;
	std::string test_text;

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];

		if (arg == "-h" || arg == "--help") {
			print_help();
			return 0;
		}
		else if (arg == "--demo") {
			demo_mode = true;
		}
		else if (arg == "--test" && i + 1 < argc) {
			test_text = argv[++i];
		}
		else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
			input_file = argv[++i];
		}
		else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
			output_prefix = argv[++i];
		}
		else if ((arg == "-v" || arg == "--vocab-size") && i + 1 < argc) {
			vocab_size = std::stoi(argv[++i]);
		}
		else if ((arg == "-m" || arg == "--min-freq") && i + 1 < argc) {
			min_frequency = std::stoi(argv[++i]);
		}
		else if (arg == "--jsonl") {
			is_jsonl = true;
		}
		else if (arg == "--text-field" && i + 1 < argc) {
			text_field = argv[++i];
		}
		else if (arg == "--no-lowercase") {
			use_lowercase = false;
		}
		else if (arg == "--no-punct-split") {
			split_punctuation = false;
		}
		else {
			std::cerr << "Unknown argument: " << arg << std::endl;
			print_help();
			return 1;
		}
	}

	// Handle different modes
	if (demo_mode) {
		run_demo();
		return 0;
	}

	if (!test_text.empty()) {
		test_tokenization(test_text);
		return 0;
	}

	if (input_file.empty()) {
		std::cout << "No input file specified. Use --demo for a quick test." << std::endl;
		print_help();
		return 1;
	}

	// Main training mode
	std::cout << "Starting BPE training..." << std::endl;
	std::cout << "   Input: " << input_file << std::endl;
	std::cout << "   Format: " << (is_jsonl ? "JSONL" : "Plain text") << std::endl;
	std::cout << "   Vocab size: " << vocab_size << std::endl;
	std::cout << "   Min frequency: " << min_frequency << std::endl;
	std::cout << "   Output prefix: " << output_prefix << std::endl;

	// Initialize trainer
	TinyBPETrainer trainer;
	trainer
		.set_lowercase(use_lowercase)
		.set_split_punctuation(split_punctuation)
		.set_normalize_whitespace(true);

	// Train BPE
	auto start_time = std::chrono::high_resolution_clock::now();

	bool success = false;
	if (is_jsonl) {
		success = trainer.train_from_jsonl(input_file, text_field, vocab_size, min_frequency);
	}
	else {
		success = trainer.train_from_file(input_file, vocab_size, min_frequency);
	}

	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

	if (success) {
		// Save outputs
		std::string vocab_file = output_prefix + "_vocab.txt";
		std::string merges_file = output_prefix + "_merges.txt";

		trainer.save_vocab(vocab_file);
		trainer.save_merges(merges_file);

		// Show final statistics
		std::cout << "\nTraining completed successfully!" << std::endl;
		std::cout << "   Total time: " << duration.count() << " seconds" << std::endl;
		trainer.print_stats();

		std::cout << "\nOutput files:" << std::endl;
		std::cout << "    " << vocab_file << " (vocabulary)" << std::endl;
		std::cout << "    " << merges_file << " (BPE merges)" << std::endl;

		// Test tokenization with some examples
		std::cout << "\nTesting trained tokenizer:" << std::endl;

		std::vector<std::string> test_sentences = {
			"Hello, world!",
			"This is a test sentence.",
			"Machine learning with transformers.",
			"Tokenization preprocessing pipeline."
		};

		for (const auto& sentence : test_sentences) {
			auto tokens = trainer.tokenize_test(sentence);
			std::cout << "  \"" << sentence << "\" → " << tokens.size() << " tokens" << std::endl;
		}

		std::cout << "\nUsage with Modern C++ Text Tokenizer:" << std::endl;
		std::cout << "   TextTokenizer tokenizer;" << std::endl;
		std::cout << "   tokenizer.load_vocab(\"" << vocab_file << "\");" << std::endl;
		std::cout << "   auto tokens = tokenizer.encode(\"your text here\");" << std::endl;

	}
	else {
		std::cout << "Training failed!" << std::endl;
		return 1;
	}

	return 0;
}
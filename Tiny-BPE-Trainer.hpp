/*
 * Tiny-BPE-Trainer.hpp
 * ---------------------------------------------------------------------------
 * Tiny BPE Trainer — A lightweight, header-only Byte Pair Encoding (BPE)
 * tokenizer trainer for modern C++17 projects.
 *
 * Author: Mecanik1337 (https://mecanik.dev/en/)
 * Repository: https://github.com/Mecanik/Tiny-BPE-Trainer
 * License: MIT License
 *
 * Description:
 *  This header-only library implements a fast, UTF-8-safe, and dependency-free
 *  BPE trainer that generates HuggingFace-compatible vocabularies and merge rules.
 *  Ideal for building custom tokenizers for transformers and NLP pipelines.
 *
 * Features:
 *  - Full BPE algorithm with configurable vocabulary size and merge threshold
 *  - Supports plain text and JSONL input formats
 *  - Outputs HuggingFace-compatible vocab.txt and merges.txt
 *  - Unicode-aware and whitespace/punctuation normalization
 *  - CLI-friendly and easily embeddable into C++ projects
 *
 * Designed for use in:
 *  - Training custom subword vocabularies for LLMs
 *  - Integrating with Mecanik's Modern Text Tokenizer
 *  - NLP preprocessing and embedded inference pipelines
 *
 * See README for usage examples, benchmarks, and integration tips.
 *
 * Contributions welcome!
 * ---------------------------------------------------------------------------
 */

#pragma once
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <regex>
#include <chrono>
#include <iomanip>

namespace MecanikDev {

	class TinyBPETrainer {
	private:
		// Core data structures
		std::unordered_map<std::string, int> token_frequencies_;
		std::vector<std::pair<std::string, std::string>> merges_;
		std::unordered_set<std::string> vocabulary_;

		// Configuration
		bool lowercase_;
		bool split_punctuation_;
		bool normalize_whitespace_;
		std::string special_tokens_[4] = { "<|endoftext|>", "<|unk|>", "<|pad|>", "<|mask|>" };

		// Statistics
		size_t total_chars_processed_ = 0;
		size_t total_words_processed_ = 0;

		// Helper functions
		std::string normalize_text(const std::string& text) const {
			std::string result = text;

			if (normalize_whitespace_) {
				// Replace multiple whitespaces with single space
				std::regex ws_regex("\\s+");
				result = std::regex_replace(result, ws_regex, " ");
			}

			if (split_punctuation_) {
				// Add spaces around punctuation
				std::regex punct_regex("([.,!?;:()\\[\\]{}\"'])");
				result = std::regex_replace(result, punct_regex, " $1 ");
			}

			if (lowercase_) {
				std::transform(result.begin(), result.end(), result.begin(), ::tolower);
			}

			return result;
		}

		std::vector<std::string> split_into_words(const std::string& text) const {
			std::vector<std::string> words;
			std::istringstream iss(text);
			std::string word;

			while (iss >> word) {
				if (!word.empty()) {
					words.push_back(word);
				}
			}

			return words;
		}

		std::vector<std::string> word_to_chars(const std::string& word) const {
			std::vector<std::string> chars;

			for (size_t i = 0; i < word.size(); ) {
				// Handle UTF-8 multibyte characters
				unsigned char c = word[i];
				size_t char_len = 1;

				if ((c & 0x80) == 0) {
					char_len = 1;  // ASCII
				}
				else if ((c & 0xE0) == 0xC0) {
					char_len = 2;  // 2-byte UTF-8
				}
				else if ((c & 0xF0) == 0xE0) {
					char_len = 3;  // 3-byte UTF-8
				}
				else if ((c & 0xF8) == 0xF0) {
					char_len = 4;  // 4-byte UTF-8
				}

				std::string character = word.substr(i, char_len);
				chars.push_back(character);
				i += char_len;
			}

			// Add end-of-word marker
			if (!chars.empty()) {
				chars.back() += "</w>";
			}

			return chars;
		}

		std::unordered_map<std::string, int> get_pairs(const std::vector<std::string>& word) const {
			std::unordered_map<std::string, int> pairs;

			for (size_t i = 0; i < word.size() - 1; ++i) {
				std::string pair = word[i] + " " + word[i + 1];
				pairs[pair]++;
			}

			return pairs;
		}

		std::string find_best_pair(const std::unordered_map<std::string, std::unordered_map<std::string, int>>& pair_counts) const {
			std::string best_pair;
			int max_count = 0;

			for (const auto& word_pairs : pair_counts) {
				for (const auto& pair_count : word_pairs.second) {
					if (pair_count.second > max_count) {
						max_count = pair_count.second;
						best_pair = pair_count.first;
					}
				}
			}

			return best_pair;
		}

		std::vector<std::string> merge_word(const std::vector<std::string>& word, const std::string& pair) const {
			std::vector<std::string> result;
			std::istringstream iss(pair);
			std::string first, second;
			iss >> first >> second;

			size_t i = 0;
			while (i < word.size()) {
				if (i < word.size() - 1 && word[i] == first && word[i + 1] == second) {
					result.push_back(first + second);
					i += 2;
				}
				else {
					result.push_back(word[i]);
					i++;
				}
			}

			return result;
		}

	public:
		TinyBPETrainer()
			: lowercase_(true)
			, split_punctuation_(true)
			, normalize_whitespace_(true) {
		}

		// Configuration methods
		TinyBPETrainer& set_lowercase(bool enable) {
			lowercase_ = enable;
			return *this;
		}

		TinyBPETrainer& set_split_punctuation(bool enable) {
			split_punctuation_ = enable;
			return *this;
		}

		TinyBPETrainer& set_normalize_whitespace(bool enable) {
			normalize_whitespace_ = enable;
			return *this;
		}

		TinyBPETrainer& set_special_tokens(const std::string& eos, const std::string& unk,
			const std::string& pad, const std::string& mask) {
			special_tokens_[0] = eos;
			special_tokens_[1] = unk;
			special_tokens_[2] = pad;
			special_tokens_[3] = mask;
			return *this;
		}

		// Training from text file
		bool train_from_file(const std::string& filepath, int vocab_size = 32000, int min_frequency = 2) {
			std::ifstream file(filepath);
			if (!file.is_open()) {
				std::cerr << "Error: Cannot open file " << filepath << std::endl;
				return false;
			}

			std::cout << "Reading corpus from: " << filepath << std::endl;

			// Build initial word frequencies
			std::unordered_map<std::vector<std::string>, int, VectorHash> word_freqs;
			std::string line;

			while (std::getline(file, line)) {
				if (line.empty()) continue;

				total_chars_processed_ += line.size();
				std::string normalized = normalize_text(line);
				auto words = split_into_words(normalized);

				for (const auto& word : words) {
					auto chars = word_to_chars(word);
					word_freqs[chars]++;
					total_words_processed_++;
				}
			}

			std::cout << "Processed " << total_chars_processed_ << " characters, "
				<< total_words_processed_ << " words" << std::endl;
			std::cout << "Unique word forms: " << word_freqs.size() << std::endl;

			// Initialize vocabulary with characters
			vocabulary_.clear();
			for (const auto& word_freq : word_freqs) {
				for (const auto& char_token : word_freq.first) {
					vocabulary_.insert(char_token);
				}
			}

			// Add special tokens
			for (const auto& special : special_tokens_) {
				vocabulary_.insert(special);
			}

			std::cout << "Initial vocabulary size: " << vocabulary_.size() << std::endl;

			// BPE Training
			std::cout << "Starting BPE training..." << std::endl;

			auto start_time = std::chrono::high_resolution_clock::now();

			while (static_cast<int>(vocabulary_.size()) < vocab_size) {
				// Count all pairs across all words
				std::unordered_map<std::string, int> pair_counts;

				for (const auto& word_freq : word_freqs) {
					if (word_freq.second < min_frequency) continue;

					auto pairs = get_pairs(word_freq.first);
					for (const auto& pair : pairs) {
						pair_counts[pair.first] += pair.second * word_freq.second;
					}
				}

				if (pair_counts.empty()) break;

				// Find most frequent pair
				std::string best_pair;
				int max_count = 0;
				for (const auto& pair : pair_counts) {
					if (pair.second > max_count) {
						max_count = pair.second;
						best_pair = pair.first;
					}
				}

				if (max_count < min_frequency) break;

				// Record the merge
				std::istringstream iss(best_pair);
				std::string first, second;
				iss >> first >> second;
				merges_.push_back({ first, second });

				// Add merged token to vocabulary
				std::string merged_token = first + second;
				vocabulary_.insert(merged_token);

				// Update word frequencies with merged pairs
				std::unordered_map<std::vector<std::string>, int, VectorHash> new_word_freqs;
				for (const auto& word_freq : word_freqs) {
					auto merged_word = merge_word(word_freq.first, best_pair);
					new_word_freqs[merged_word] = word_freq.second;
				}
				word_freqs = std::move(new_word_freqs);

				if (merges_.size() % 1000 == 0) {
					std::cout << "  Merge " << merges_.size() << ": '" << first << "' + '"
						<< second << "' → '" << merged_token << "' (freq: " << max_count << ")" << std::endl;
				}
			}

			auto end_time = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

			std::cout << "BPE training completed!" << std::endl;
			std::cout << "   Final vocabulary size: " << vocabulary_.size() << std::endl;
			std::cout << "   Total merges: " << merges_.size() << std::endl;
			std::cout << "   Training time: " << duration.count() << " seconds" << std::endl;

			return true;
		}

		// Training from JSONL file (common format for datasets)
		bool train_from_jsonl(const std::string& filepath, const std::string& text_field = "text",
			int vocab_size = 32000, int min_frequency = 2) {
			std::ifstream file(filepath);
			if (!file.is_open()) {
				std::cerr << "Error: Cannot open JSONL file " << filepath << std::endl;
				return false;
			}

			std::cout << "Reading JSONL corpus from: " << filepath << std::endl;
			std::cout << "Looking for field: '" << text_field << "'" << std::endl;

			// Simple JSON parsing (for text field only)
			std::unordered_map<std::vector<std::string>, int, VectorHash> word_freqs;
			std::string line;
			int line_count = 0;

			while (std::getline(file, line)) {
				if (line.empty()) continue;

				line_count++;

				// Simple JSON text extraction
				std::string field_pattern = "\"" + text_field + "\":";
				size_t field_pos = line.find(field_pattern);
				if (field_pos == std::string::npos) continue;

				size_t text_start = line.find("\"", field_pos + field_pattern.length());
				if (text_start == std::string::npos) continue;
				text_start++;

				size_t text_end = line.find("\"", text_start);
				if (text_end == std::string::npos) continue;

				std::string text_content = line.substr(text_start, text_end - text_start);

				// Process the extracted text
				total_chars_processed_ += text_content.size();
				std::string normalized = normalize_text(text_content);
				auto words = split_into_words(normalized);

				for (const auto& word : words) {
					auto chars = word_to_chars(word);
					word_freqs[chars]++;
					total_words_processed_++;
				}

				if (line_count % 10000 == 0) {
					std::cout << "  Processed " << line_count << " lines..." << std::endl;
				}
			}

			std::cout << "Processing complete:" << std::endl;
			std::cout << "   Lines processed: " << line_count << std::endl;
			std::cout << "   Characters: " << total_chars_processed_ << std::endl;
			std::cout << "   Words: " << total_words_processed_ << std::endl;
			std::cout << "   Unique word forms: " << word_freqs.size() << std::endl;

			// Continue with regular BPE training (same as train_from_file)
			return train_bpe_from_word_freqs(word_freqs, vocab_size, min_frequency);
		}

		// Save vocabulary in HuggingFace format
		bool save_vocab(const std::string& vocab_path) const {
			std::ofstream file(vocab_path);
			if (!file.is_open()) {
				std::cerr << "Error: Cannot create vocab file " << vocab_path << std::endl;
				return false;
			}

			// First add special tokens
			for (const auto& special : special_tokens_) {
				file << special << "\n";
			}

			// Then add all other tokens sorted by frequency/alphabetically
			std::vector<std::string> sorted_vocab;
			for (const auto& token : vocabulary_) {
				bool is_special = false;
				for (const auto& special : special_tokens_) {
					if (token == special) {
						is_special = true;
						break;
					}
				}
				if (!is_special) {
					sorted_vocab.push_back(token);
				}
			}

			std::sort(sorted_vocab.begin(), sorted_vocab.end());

			for (const auto& token : sorted_vocab) {
				file << token << "\n";
			}

			std::cout << "Saved vocabulary (" << vocabulary_.size() << " tokens) to: " << vocab_path << std::endl;
			return true;
		}

		// Save merges in HuggingFace format
		bool save_merges(const std::string& merges_path) const {
			std::ofstream file(merges_path);
			if (!file.is_open()) {
				std::cerr << "Error: Cannot create merges file " << merges_path << std::endl;
				return false;
			}

			file << "#version: 0.2\n";

			for (const auto& merge : merges_) {
				file << merge.first << " " << merge.second << "\n";
			}

			std::cout << "Saved merges (" << merges_.size() << " rules) to: " << merges_path << std::endl;
			return true;
		}

		// Get training statistics
		void print_stats() const {
			std::cout << "\nTraining Statistics:" << std::endl;
			std::cout << "   Characters processed: " << total_chars_processed_ << std::endl;
			std::cout << "   Words processed: " << total_words_processed_ << std::endl;
			std::cout << "   Final vocab size: " << vocabulary_.size() << std::endl;
			std::cout << "   BPE merges: " << merges_.size() << std::endl;

			if (total_chars_processed_ > 0) {
				double compression_ratio = static_cast<double>(vocabulary_.size()) / total_chars_processed_;
				std::cout << "   Compression ratio: " << std::fixed << std::setprecision(4)
					<< compression_ratio << std::endl;
			}
		}

		// Test tokenization with trained BPE
		std::vector<std::string> tokenize_test(const std::string& text) const {
			std::string normalized = normalize_text(text);
			auto words = split_into_words(normalized);
			std::vector<std::string> result;

			for (const auto& word : words) {
				auto chars = word_to_chars(word);

				// Apply all merges in order
				for (const auto& merge : merges_) {
					std::string pair = merge.first + " " + merge.second;
					chars = merge_word(chars, pair);
				}

				result.insert(result.end(), chars.begin(), chars.end());
			}

			return result;
		}

	private:
		// Helper struct for hashing vectors (needed for unordered_map with vector keys)
		struct VectorHash {
			size_t operator()(const std::vector<std::string>& v) const {
				size_t seed = v.size();
				for (auto& i : v) {
					seed ^= std::hash<std::string>{}(i)+0x9e3779b9 + (seed << 6) + (seed >> 2);
				}
				return seed;
			}
		};

		// Extracted BPE training logic for code reuse
		bool train_bpe_from_word_freqs(const std::unordered_map<std::vector<std::string>, int, VectorHash>& initial_word_freqs,
			int vocab_size, int min_frequency) {
			auto word_freqs = initial_word_freqs;

			// Initialize vocabulary
			vocabulary_.clear();
			for (const auto& word_freq : word_freqs) {
				for (const auto& char_token : word_freq.first) {
					vocabulary_.insert(char_token);
				}
			}

			for (const auto& special : special_tokens_) {
				vocabulary_.insert(special);
			}

			std::cout << "Initial vocabulary size: " << vocabulary_.size() << std::endl;
			std::cout << "Starting BPE training..." << std::endl;

			auto start_time = std::chrono::high_resolution_clock::now();

			while (static_cast<int>(vocabulary_.size()) < vocab_size) {
				std::unordered_map<std::string, int> pair_counts;

				for (const auto& word_freq : word_freqs) {
					if (word_freq.second < min_frequency) continue;

					auto pairs = get_pairs(word_freq.first);
					for (const auto& pair : pairs) {
						pair_counts[pair.first] += pair.second * word_freq.second;
					}
				}

				if (pair_counts.empty()) break;

				std::string best_pair;
				int max_count = 0;
				for (const auto& pair : pair_counts) {
					if (pair.second > max_count) {
						max_count = pair.second;
						best_pair = pair.first;
					}
				}

				if (max_count < min_frequency) break;

				std::istringstream iss(best_pair);
				std::string first, second;
				iss >> first >> second;
				merges_.push_back({ first, second });

				std::string merged_token = first + second;
				vocabulary_.insert(merged_token);

				std::unordered_map<std::vector<std::string>, int, VectorHash> new_word_freqs;
				for (const auto& word_freq : word_freqs) {
					auto merged_word = merge_word(word_freq.first, best_pair);
					new_word_freqs[merged_word] = word_freq.second;
				}
				word_freqs = std::move(new_word_freqs);

				if (merges_.size() % 1000 == 0) {
					std::cout << "  Merge " << merges_.size() << ": '" << first << "' + '"
						<< second << "' → '" << merged_token << "' (freq: " << max_count << ")" << std::endl;
				}
			}

			auto end_time = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

			std::cout << "BPE training completed in " << duration.count() << " seconds!" << std::endl;
			std::cout << "   Final vocabulary size: " << vocabulary_.size() << std::endl;
			std::cout << "   Total merges: " << merges_.size() << std::endl;

			return true;
		}
	};

} // namespace MecanikDev

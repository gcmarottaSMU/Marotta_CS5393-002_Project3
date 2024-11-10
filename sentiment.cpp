#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <omp.h>
#include <functional>

class DSString {
private:
    char* data;
    int len;

    void copyData(const char* str) {
        if (str) {
            len = strlen(str);
            data = new char[len + 1];
            strcpy(data, str);
        } else {
            data = nullptr;
            len = 0;
        }
    }

public:
    DSString() : data(nullptr), len(0) {}
    DSString(const char* str) {
        copyData(str);
    }
    DSString(const DSString& other) {
        copyData(other.data);
    }
    ~DSString() {
        delete[] data;
    }

    DSString& operator=(const DSString& other) {
        if (this != &other) {
            delete[] data;
            copyData(other.data);
        }
        return *this;
    }

    DSString& operator=(const char* str) {
        delete[] data;
        copyData(str);
        return *this;
    }

    bool operator==(const DSString& other) const {
        return strcmp(data, other.data) == 0;
    }

    bool operator!=(const DSString& other) const {
        return !(*this == other);
    }

    bool operator<(const DSString& other) const {
        return strcmp(data, other.data) < 0;
    }

    DSString operator+(const DSString& other) const {
        char* concatenated = new char[len + other.len + 1];
        strcpy(concatenated, data);
        strcat(concatenated, other.data);
        DSString result(concatenated);
        delete[] concatenated;
        return result;
    }

    char& operator[](int index) {
        if (index < 0 || index >= len) {
            throw std::out_of_range("Index out of range");
        }
        return data[index];
    }

    const char* c_str() const {
        return data ? data : "";
    }

    int length() const {
        return len;
    }

    // Utility Functions
    DSString substr(int start, int length) const {
        if (start < 0 || start >= len) {
            throw std::out_of_range("Start index out of range");
        }
        if (start + length > len) {
            length = len - start;
        }
        char* sub = new char[length + 1];
        strncpy(sub, data + start, length);
        sub[length] = '\0';
        DSString result(sub);
        delete[] sub;
        return result;
    }

    int find(const DSString& substr) const {
        if (!data || !substr.data) return -1;
        const char* pos = strstr(data, substr.c_str());
        if (pos) {
            return pos - data;
        }
        return -1;
    }

    void clear() {
        delete[] data;
        data = nullptr;
        len = 0;
    }
};

namespace std {
    template <>
    struct hash<DSString> {
        size_t operator()(const DSString& s) const noexcept {
            return hash<string>()(s.c_str());
        }
    };
}

class SentimentClassifier {
public:
    void train(const std::string& trainingFile, const std::string& stopWordsFile);
    void predict(const std::string& testingFile, const std::string& resultsFile);
    void evaluatePredictions(const std::string& groundTruthFile, const std::string& resultsFile, const std::string& accuracyFile);

private:
    std::unordered_map<DSString, int> wordSentiment; // Positive count if value > 0, negative if < 0
    std::unordered_set<DSString> stopWords; // Set of stop words to ignore during tokenization
    void tokenize(const DSString& tweet, std::vector<DSString>& words);
    void loadStopWords(const std::string& stopWordsFile);
    DSString stem(const DSString& word); // Simple stemmer
};

void SentimentClassifier::loadStopWords(const std::string& stopWordsFile) {
    std::ifstream file(stopWordsFile);
    if (!file.is_open()) {
        std::cerr << "Error opening stop words file" << std::endl;
        return;
    }

    std::string word;
    while (std::getline(file, word)) {
        // Remove possible punctuation, convert to lowercase
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        if (!word.empty()) {
            stopWords.insert(DSString(word.c_str()));
        }
    }
    file.close();
}

DSString SentimentClassifier::stem(const DSString& word) {
    // Placeholder for a stemming algorithm (e.g., Porter Stemmer)
    // For demonstration, return the word as-is
    // Implement or integrate a full stemming algorithm for better performance
    return DSString(word.c_str());
}

void SentimentClassifier::tokenize(const DSString& tweet, std::vector<DSString>& words) {
    std::string tweetStr = tweet.c_str();
    std::transform(tweetStr.begin(), tweetStr.end(), tweetStr.begin(), ::tolower);

    std::stringstream ss(tweetStr);
    std::string word;
    while (ss >> word) {
        // Remove punctuation from the word
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
        // Apply stemming
        DSString stemmedWord = stem(DSString(word.c_str()));
        if (!stemmedWord.c_str()[0] == '\0') { // Ensure the word is not empty
            words.emplace_back(stemmedWord);
        }
    }
}

void SentimentClassifier::train(const std::string& trainingFile, const std::string& stopWordsFile) {
    loadStopWords(stopWordsFile);
    
    std::ifstream file(trainingFile);
    if (!file.is_open()) {
        std::cerr << "Error opening training file" << std::endl;
        return;
    }

    std::vector<std::string> lines;
    std::string line;
    // Skip header line
    if (!std::getline(file, line)) {
        std::cerr << "Training file is empty." << std::endl;
        return;
    }

    // Read all lines into a vector for parallel processing
    while (std::getline(file, line)) {
        lines.emplace_back(line);
    }
    file.close();

    #pragma omp parallel for
    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string& currentLine = lines[i];
        std::stringstream ss(currentLine);
        std::string sentimentStr, id, date, query, user, tweet;
        std::getline(ss, sentimentStr, ',');
        std::getline(ss, id, ',');
        std::getline(ss, date, ',');
        std::getline(ss, query, ',');
        std::getline(ss, user, ',');
        std::getline(ss, tweet);

        int sentiment;
        try {
            sentiment = std::stoi(sentimentStr);
        } catch (const std::invalid_argument& e) {
            #pragma omp critical
            std::cerr << "Invalid sentiment value: " << sentimentStr << " in line: " << currentLine << std::endl;
            continue;
        } catch (const std::out_of_range& e) {
            #pragma omp critical
            std::cerr << "Out of range sentiment value: " << sentimentStr << " in line: " << currentLine << std::endl;
            continue;
        }

        DSString tweetDS(tweet.c_str());
        std::vector<DSString> words;
        tokenize(tweetDS, words);

        for (const DSString& word : words) {
            if (stopWords.find(word) == stopWords.end()) {
                #pragma omp atomic
                wordSentiment[word] += (sentiment == 4) ? 1 : -1;
            }
        }
    }

    std::cout << "Training completed. Vocabulary size: " << wordSentiment.size() << std::endl;
}

void SentimentClassifier::predict(const std::string& testingFile, const std::string& resultsFile) {
    std::ifstream file(testingFile);
    if (!file.is_open()) {
        std::cerr << "Error opening testing file" << std::endl;
        return;
    }

    std::ofstream results(resultsFile);
    if (!results.is_open()) {
        std::cerr << "Error opening results file" << std::endl;
        return;
    }

    std::string line;
    // Read header if present
    if (!std::getline(file, line)) {
        std::cerr << "Testing file is empty." << std::endl;
        return;
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string id, date, query, user, tweet;
        std::getline(ss, id, ',');
        std::getline(ss, date, ',');
        std::getline(ss, query, ',');
        std::getline(ss, user, ',');
        std::getline(ss, tweet);

        DSString tweetDS(tweet.c_str());
        std::vector<DSString> words;
        tokenize(tweetDS, words);

        int sentimentScore = 0;
        for (const DSString& word : words) {
            if (stopWords.find(word) == stopWords.end()) {
                auto it = wordSentiment.find(word);
                if (it != wordSentiment.end()) {
                    sentimentScore += it->second;
                }
            }
        }

        int predictedSentiment = (sentimentScore >= 0) ? 4 : 0;
        results << predictedSentiment << ", " << id << std::endl;
    }
    file.close();
    results.close();

    std::cout << "Prediction completed. Results saved to " << resultsFile << std::endl;
}

void SentimentClassifier::evaluatePredictions(const std::string& groundTruthFile, const std::string& resultsFile, const std::string& accuracyFile) {
    std::ifstream groundTruth(groundTruthFile);
    std::ifstream results(resultsFile);
    if (!groundTruth.is_open()) {
        std::cerr << "Error opening ground truth file" << std::endl;
        return;
    }
    if (!results.is_open()) {
        std::cerr << "Error opening results file" << std::endl;
        return;
    }

    std::ofstream accuracy(accuracyFile);
    if (!accuracy.is_open()) {
        std::cerr << "Error opening accuracy file" << std::endl;
        return;
    }

    int totalTweets = 0;
    int correctPredictions = 0;
    std::string truthLine, resultLine;

    // Read headers if present
    std::getline(groundTruth, truthLine);
    std::getline(results, resultLine);

    while (std::getline(groundTruth, truthLine) && std::getline(results, resultLine)) {
        std::stringstream truthSS(truthLine);
        std::stringstream resultSS(resultLine);
        std::string truthSentiment, truthId;
        std::string resultSentiment, resultId;

        std::getline(truthSS, truthSentiment, ',');
        std::getline(truthSS, truthId, ',');
        std::getline(resultSS, resultSentiment, ',');
        std::getline(resultSS, resultId, ',');

        if (truthSentiment == resultSentiment) {
            ++correctPredictions;
        } else {
            accuracy << "ID: " << truthId 
                     << ", Actual: " << truthSentiment 
                     << ", Predicted: " << resultSentiment 
                     << std::endl;
        }
        ++totalTweets;
    }

    double accuracyValue = (totalTweets > 0) ? static_cast<double>(correctPredictions) / totalTweets : 0.0;
    accuracy << "Overall Accuracy: " << std::fixed << std::setprecision(3) << accuracyValue << std::endl;

    groundTruth.close();
    results.close();
    accuracy.close();

    std::cout << "Evaluation completed. Accuracy saved to " << accuracyFile << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 7) { // Expecting 6 arguments + program name
        std::cerr << "Usage: ./sentiment <trainingFile> <stopWordsFile> <testingFile> <groundTruthFile> <resultsFile> <accuracyFile>" << std::endl;
        return 1;
    }

    std::string trainingFile = argv[1];
    std::string stopWordsFile = argv[2];
    std::string testingFile = argv[3];
    std::string groundTruthFile = argv[4];
    std::string resultsFile = argv[5];
    std::string accuracyFile = argv[6];

    SentimentClassifier classifier;
    classifier.train(trainingFile, stopWordsFile);
    classifier.predict(testingFile, resultsFile);
    classifier.evaluatePredictions(groundTruthFile, resultsFile, accuracyFile);

    return 0;
}

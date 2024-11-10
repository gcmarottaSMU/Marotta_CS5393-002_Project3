#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <tuple>

// ----------------------- DSString Class -----------------------
class DSString {
private:
    char* data;
    size_t len;

    // Helper function to copy data
    void copyData(const char* str) {
        if (str) {
            len = 0;
            while (str[len] != '\0') len++;
            data = new char[len + 1];
            for (size_t i = 0; i < len; ++i) {
                data[i] = str[i];
            }
            data[len] = '\0';
        }
        else {
            data = nullptr;
            len = 0;
        }
    }

public:
    // Constructors
    DSString() : data(nullptr), len(0) {}

    DSString(const char* str) {
        copyData(str);
    }

    DSString(const DSString& other) {
        copyData(other.data);
    }

    // Destructor
    ~DSString() {
        delete[] data;
    }

    // Assignment operator
    DSString& operator=(const DSString& other) {
        if (this != &other) {
            delete[] data;
            copyData(other.data);
        }
        return *this;
    }

    // Equality operator
    bool operator==(const DSString& other) const {
        if (len != other.len)
            return false;
        for (size_t i = 0; i < len; ++i) {
            if (data[i] != other.data[i])
                return false;
        }
        return true;
    }

    // Less-than operator for sorting
    bool operator<(const DSString& other) const {
        size_t minLen = (len < other.len) ? len : other.len;
        for (size_t i = 0; i < minLen; ++i) {
            if (data[i] < other.data[i])
                return true;
            if (data[i] > other.data[i])
                return false;
        }
        return len < other.len;
    }

    // Concatenation operator
    DSString operator+(const DSString& other) const {
        size_t newLen = len + other.len;
        char* concatenated = new char[newLen + 1];
        for (size_t i = 0; i < len; ++i)
            concatenated[i] = data[i];
        for (size_t i = 0; i < other.len; ++i)
            concatenated[len + i] = other.data[i];
        concatenated[newLen] = '\0';
        DSString result(concatenated);
        delete[] concatenated;
        return result;
    }

    // Access operators
    char& operator[](size_t index) {
        if (index >= len)
            throw std::out_of_range("Index out of range");
        return data[index];
    }

    const char& operator[](size_t index) const {
        if (index >= len)
            throw std::out_of_range("Index out of range");
        return data[index];
    }

    // c_str function
    const char* c_str() const {
        return data ? data : "";
    }

    // length function
    size_t length() const {
        return len;
    }

    // Substring
    DSString substr(size_t start, size_t length) const {
        if (start >= len)
            throw std::out_of_range("Start index out of range");
        if (start + length > len)
            length = len - start;
        char* sub = new char[length + 1];
        for (size_t i = 0; i < length; ++i)
            sub[i] = data[start + i];
        sub[length] = '\0';
        DSString result(sub);
        delete[] sub;
        return result;
    }

    // Find function
    int find(const DSString& substr) const {
        if (substr.len == 0)
            return 0;
        if (substr.len > len)
            return -1;
        for (size_t i = 0; i <= len - substr.len; ++i) {
            bool match = true;
            for (size_t j = 0; j < substr.len; ++j) {
                if (data[i + j] != substr.data[j]) {
                    match = false;
                    break;
                }
            }
            if (match)
                return static_cast<int>(i);
        }
        return -1;
    }

    // Clear function
    void clear() {
        delete[] data;
        data = nullptr;
        len = 0;
    }
};

// Hash specialization for DSString
namespace std {
    template <>
    struct hash<DSString> {
        size_t operator()(const DSString& s) const {
            size_t hashVal = 0;
            for (size_t i = 0; i < s.length(); ++i) {
                hashVal = hashVal * 31 + static_cast<size_t>(s.c_str()[i]);
            }
            return hashVal;
        }
    };
}

// ------------------- SentimentClassifier Class -------------------
class SentimentClassifier {
public:
    // Training, Prediction, and Evaluation functions
    void train(const std::string& trainingFile);
    void predict(const std::string& testingFile, const std::string& resultsFile);
    void evaluatePredictions(const std::string& groundTruthFile, const std::string& resultsFile, const std::string& accuracyFile);

private:
    std::unordered_map<DSString, int> wordSentiment; // Positive count if value > 0, negative if < 0
    std::unordered_set<DSString> stopWords; // Set of stop words to ignore during tokenization

    // Helper functions
    void loadStopWords(); // Load a predefined set of stop words
    void tokenize(const DSString& tweet, std::vector<DSString>& words);
    DSString toLower(const DSString& word);
};

// Load a predefined set of stop words
void SentimentClassifier::loadStopWords() {
    // A minimal set of English stop words. For a comprehensive list, consider expanding this.
    std::vector<std::string> stopWordsList = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by",
        "for", "if", "in", "into", "is", "it",
        "no", "not", "of", "on", "or", "such",
        "that", "the", "their", "then", "there", "these",
        "they", "this", "to", "was", "will", "with"
    };

    for (const auto& word : stopWordsList) {
        DSString dsWord(word.c_str());
        stopWords.insert(dsWord);
    }
}

// Convert DSString to lowercase
DSString SentimentClassifier::toLower(const DSString& word) {
    std::string temp = word.c_str();
    std::transform(temp.begin(), temp.end(), temp.begin(), ::tolower);
    return DSString(temp.c_str());
}

// Tokenize a tweet into words, removing stop words and punctuation
void SentimentClassifier::tokenize(const DSString& tweet, std::vector<DSString>& words) {
    std::string tweetStr = tweet.c_str();
    std::transform(tweetStr.begin(), tweetStr.end(), tweetStr.begin(), ::tolower);

    std::stringstream ss(tweetStr);
    std::string word;
    while (ss >> word) {
        // Remove punctuation from the word
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
        if (word.empty())
            continue;
        DSString dsWord(word.c_str());
        if (stopWords.find(dsWord) == stopWords.end()) { // If not a stop word
            words.emplace_back(dsWord);
        }
    }
}

// Training function
void SentimentClassifier::train(const std::string& trainingFile) {
    loadStopWords();

    std::ifstream file(trainingFile);
    if (!file.is_open()) {
        std::cerr << "Error opening training file: " << trainingFile << std::endl;
        exit(1);
    }

    std::string line;
    // No header line based on assignment examples
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string sentimentStr, id, date, query, user, tweet;

        // Parse CSV line
        if (!std::getline(ss, sentimentStr, ',')) continue;
        if (!std::getline(ss, id, ',')) continue;
        if (!std::getline(ss, date, ',')) continue;
        if (!std::getline(ss, query, ',')) continue;
        if (!std::getline(ss, user, ',')) continue;
        if (!std::getline(ss, tweet)) continue;

        int sentiment;
        try {
            sentiment = std::stoi(sentimentStr);
        }
        catch (...) {
            // Invalid sentiment value
            continue;
        }

        if (sentiment != 0 && sentiment != 4)
            continue; // Ignore sentiments not 0 or 4

        DSString tweetDS(tweet.c_str());
        std::vector<DSString> words;
        tokenize(tweetDS, words);

        for (const auto& word : words) {
            if (sentiment == 4)
                wordSentiment[word] += 1; // Positive
            else
                wordSentiment[word] -= 1; // Negative
        }
    }

    file.close();
    std::cout << "Training completed. Vocabulary size: " << wordSentiment.size() << std::endl;
}

// Prediction function
void SentimentClassifier::predict(const std::string& testingFile, const std::string& resultsFile) {
    std::ifstream file(testingFile);
    if (!file.is_open()) {
        std::cerr << "Error opening testing file: " << testingFile << std::endl;
        exit(1);
    }

    std::ofstream results(resultsFile);
    if (!results.is_open()) {
        std::cerr << "Error opening results file: " << resultsFile << std::endl;
        exit(1);
    }

    std::string line;
    // No header line based on assignment examples
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string id, date, query, user, tweet;
        if (!std::getline(ss, id, ',')) continue;
        if (!std::getline(ss, date, ',')) continue;
        if (!std::getline(ss, query, ',')) continue;
        if (!std::getline(ss, user, ',')) continue;
        if (!std::getline(ss, tweet)) continue;

        DSString tweetDS(tweet.c_str());
        std::vector<DSString> words;
        tokenize(tweetDS, words);

        int sentimentScore = 0;
        for (const auto& word : words) {
            auto it = wordSentiment.find(word);
            if (it != wordSentiment.end()) {
                sentimentScore += it->second;
            }
        }

        int predictedSentiment = (sentimentScore >= 0) ? 4 : 0;
        results << predictedSentiment << ", " << id << std::endl;
    }

    file.close();
    results.close();
    std::cout << "Prediction completed. Results saved to " << resultsFile << std::endl;
}

// Evaluation function
void SentimentClassifier::evaluatePredictions(const std::string& groundTruthFile, const std::string& resultsFile, const std::string& accuracyFile) {
    std::ifstream groundTruth(groundTruthFile);
    std::ifstream results(resultsFile);
    if (!groundTruth.is_open()) {
        std::cerr << "Error opening ground truth file: " << groundTruthFile << std::endl;
        exit(1);
    }
    if (!results.is_open()) {
        std::cerr << "Error opening results file: " << resultsFile << std::endl;
        exit(1);
    }

    std::ofstream accuracyOut(accuracyFile);
    if (!accuracyOut.is_open()) {
        std::cerr << "Error opening accuracy file: " << accuracyFile << std::endl;
        exit(1);
    }

    // Read ground truth into a map
    std::unordered_map<long, int> groundTruthMap;
    std::string truthLine;
    while (std::getline(groundTruth, truthLine)) {
        std::stringstream ss(truthLine);
        std::string sentimentStr, idStr;
        if (!std::getline(ss, sentimentStr, ',')) continue;
        if (!std::getline(ss, idStr, ',')) continue;
        int sentiment;
        long tweetID;
        try {
            sentiment = std::stoi(sentimentStr);
            tweetID = std::stol(idStr);
        }
        catch (...) {
            continue; // Invalid data
        }
        groundTruthMap[tweetID] = sentiment;
    }
    groundTruth.close();

    // Read predictions
    std::vector<std::pair<int, long>> predictions; // pair<predicted sentiment, tweetID>
    std::string resultLine;
    while (std::getline(results, resultLine)) {
        std::stringstream ss(resultLine);
        std::string sentimentStr, idStr;
        if (!std::getline(ss, sentimentStr, ',')) continue;
        if (!std::getline(ss, idStr, ',')) continue;
        int sentiment;
        long tweetID;
        try {
            sentiment = std::stoi(sentimentStr);
            // Remove possible leading/trailing spaces from idStr
            size_t start = idStr.find_first_not_of(" \t");
            size_t end = idStr.find_last_not_of(" \t");
            std::string trimmedId = (start == std::string::npos) ? "" : idStr.substr(start, end - start + 1);
            tweetID = std::stol(trimmedId);
        }
        catch (...) {
            continue; // Invalid data
        }
        predictions.emplace_back(sentiment, tweetID);
    }
    results.close();

    // Compare predictions with ground truth
    int totalTweets = 0;
    int correctPredictions = 0;
    std::vector<std::tuple<int, int, long>> misclassifications; // ground truth, prediction, tweet ID

    for (const auto& pred : predictions) {
        int predicted = pred.first;
        long tweetID = pred.second;
        auto it = groundTruthMap.find(tweetID);
        if (it != groundTruthMap.end()) {
            int actual = it->second;
            if (predicted == actual) {
                correctPredictions++;
            }
            else {
                misclassifications.emplace_back(actual, predicted, tweetID);
            }
            totalTweets++;
        }
    }

    // Calculate accuracy
    double accuracyValue = (totalTweets > 0) ? (static_cast<double>(correctPredictions) / totalTweets) * 100.0 : 0.0;

    // Write accuracy first
    accuracyOut << std::fixed << std::setprecision(3) << accuracyValue << std::endl;

    // Write misclassifications
    for (const auto& mis : misclassifications) {
        accuracyOut << std::get<0>(mis) << ", " << std::get<2>(mis) << std::endl;
    }

    accuracyOut.close();
    std::cout << "Evaluation completed. Accuracy saved to " << accuracyFile << std::endl;
}

// --------------------------- Main Function ---------------------------
int main(int argc, char* argv[]) {
    if (argc != 6) { // Expecting 5 arguments + program name
        std::cerr << "Usage: ./sentiment <trainingFile> <testingFile> <groundTruthFile> <resultsFile> <accuracyFile>" << std::endl;
        return 1;
    }

    std::string trainingFile = argv[1];
    std::string testingFile = argv[2];
    std::string groundTruthFile = argv[3];
    std::string resultsFile = argv[4];
    std::string accuracyFile = argv[5];

    SentimentClassifier classifier;
    classifier.train(trainingFile);
    classifier.predict(testingFile, resultsFile);
    classifier.evaluatePredictions(groundTruthFile, resultsFile, accuracyFile);

    return 0;
}

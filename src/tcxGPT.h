#pragma once

// =============================================================================
// tcxGPT - OpenAI Responses API client for TrussC
// =============================================================================
// Header-only async GPT client using tcxCurl.
//
// Usage:
//   #include <TrussC.h>
//   #include <tcxGPT.h>
//   using namespace tcx;
//
//   GPT gpt;
//   gpt.setup(apiKey);
//   gpt.sendMessage("Hello!");
//   // In update():
//   if (gpt.hasNewResponse()) {
//       auto res = gpt.getNextResponse();
//       cout << res.text << endl;
//   }
// =============================================================================

#include <string>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <tcxCurl.h>

namespace tcx {

class GPT {
public:
    // Error codes
    enum ErrorCode {
        Success = 0,
        InvalidAPIKey,
        NetworkError,
        ServerError,
        RateLimitExceeded,
        TokenLimitExceeded,
        InvalidModel,
        BadRequest,
        Timeout,
        JSONParseError,
        UnknownError
    };

    // Response status
    enum ResponseStatus {
        InProgress,
        Completed,
        Failed,
        Incomplete
    };

    // Request data
    struct Request {
        std::string input;              // Simple text input
        nlohmann::json inputJson;       // Structured input (array with images etc.)
        std::string instructions;
        std::string model;
        float temperature;
        std::string conversationId;
        std::string previousResponseId;
        nlohmann::json metadata;
        nlohmann::json responseFormat;

        Request()
            : model("gpt-4o")
            , temperature(1.0f) {}
    };

    // Response data
    struct Response {
        std::string id;
        std::string text;
        ResponseStatus status;
        ErrorCode errorCode;
        std::string errorMessage;
        nlohmann::json fullResponse;
        int inputTokens;
        int outputTokens;
        int totalTokens;
        std::string model;

        Response()
            : status(InProgress)
            , errorCode(Success)
            , inputTokens(0)
            , outputTokens(0)
            , totalTokens(0) {}
    };

    GPT() = default;

    ~GPT() {
        running_ = false;
        if (workerThread_.joinable()) {
            workerThread_.join();
        }
    }

    // Setup with API key - starts worker thread
    void setup(const std::string& apiKey) {
        apiKey_ = apiKey;

        http_.setBaseUrl("https://api.openai.com");
        http_.setBearerToken(apiKey_);

        running_ = true;
        workerThread_ = std::thread(&GPT::workerFunction, this);
    }

    // Send a message with default settings
    void sendMessage(const std::string& input) {
        Request request;
        request.input = input;
        request.model = defaultModel_;
        request.temperature = defaultTemperature_;
        request.instructions = defaultInstructions_;
        request.conversationId = currentConversationId_;
        request.previousResponseId = lastResponseId_;
        sendRequest(request);
    }

    // Send a custom request
    void sendRequest(const Request& request) {
        std::lock_guard<std::mutex> lock(requestMutex_);
        requestQueue_.push_back(request);
    }

    // Check if there are new responses available
    bool hasNewResponse() {
        std::lock_guard<std::mutex> lock(responseMutex_);
        return !responseQueue_.empty();
    }

    // Get the next response from the queue
    Response getNextResponse() {
        std::lock_guard<std::mutex> lock(responseMutex_);
        Response response;
        if (!responseQueue_.empty()) {
            response = responseQueue_.front();
            responseQueue_.pop_front();
        }
        return response;
    }

    int getResponseQueueSize() {
        std::lock_guard<std::mutex> lock(responseMutex_);
        return static_cast<int>(responseQueue_.size());
    }

    int getRequestQueueSize() {
        std::lock_guard<std::mutex> lock(requestMutex_);
        return static_cast<int>(requestQueue_.size());
    }

    void clearRequestQueue() {
        std::lock_guard<std::mutex> lock(requestMutex_);
        requestQueue_.clear();
    }

    void clearResponseQueue() {
        std::lock_guard<std::mutex> lock(responseMutex_);
        responseQueue_.clear();
    }

    // Configuration
    void setModel(const std::string& model) { defaultModel_ = model; }
    std::string getModel() const { return defaultModel_; }

    void setTemperature(float temp) { defaultTemperature_ = std::clamp(temp, 0.0f, 2.0f); }
    float getTemperature() const { return defaultTemperature_; }

    void setInstructions(const std::string& instructions) { defaultInstructions_ = instructions; }
    std::string getInstructions() const { return defaultInstructions_; }

    void setTimeoutSeconds(int seconds) { timeoutSeconds_ = seconds; }
    int getTimeoutSeconds() const { return timeoutSeconds_; }

    void setVerbose(bool v) { verbose_ = v; }
    bool getVerbose() const { return verbose_; }

    // Conversation management
    void setConversationId(const std::string& id) { currentConversationId_ = id; }
    std::string getConversationId() const { return currentConversationId_; }

    void clearConversation() {
        currentConversationId_ = "";
        lastResponseId_ = "";
    }

    // Error message helper
    static std::string getErrorMessage(ErrorCode code) {
        switch (code) {
            case Success:            return "Success";
            case InvalidAPIKey:      return "Invalid API key";
            case NetworkError:       return "Network error";
            case ServerError:        return "Server error";
            case RateLimitExceeded:  return "Rate limit exceeded";
            case TokenLimitExceeded: return "Token limit exceeded";
            case InvalidModel:       return "Invalid model";
            case BadRequest:         return "Bad request";
            case Timeout:            return "Timeout";
            case JSONParseError:     return "JSON parse error";
            default:                 return "Unknown error";
        }
    }

private:
    // Worker thread function
    void workerFunction() {
        while (running_) {
            Request request;
            bool hasRequest = false;

            {
                std::lock_guard<std::mutex> lock(requestMutex_);
                if (!requestQueue_.empty()) {
                    request = requestQueue_.front();
                    requestQueue_.pop_front();
                    hasRequest = true;
                }
            }

            if (hasRequest) {
                Response response = processRequest(request);

                {
                    std::lock_guard<std::mutex> lock(responseMutex_);
                    responseQueue_.push_back(response);
                }

                // Update conversation state if successful
                if (response.errorCode == Success && !response.id.empty()) {
                    lastResponseId_ = response.id;
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
    }

    // Process a single request
    Response processRequest(const Request& request) {
        Response response;

        try {
            nlohmann::json body = buildRequestBody(request);

            auto res = http_.post("/v1/responses", body);

            if (!res.ok()) {
                // Handle error
                try {
                    auto j = res.json();
                    response.fullResponse = j;
                    response.errorCode = parseErrorCode(res.statusCode, j);
                    response.errorMessage = getErrorMessage(response.errorCode);

                    if (j.contains("error") && j["error"].contains("message")) {
                        response.errorMessage += ": " + j["error"]["message"].get<std::string>();
                    }
                } catch (...) {
                    response.errorCode = parseErrorCode(res.statusCode, nlohmann::json());
                    response.errorMessage = getErrorMessage(response.errorCode);
                }
                response.status = Failed;

                if (!res.error.empty()) {
                    if (res.error.find("Timeout") != std::string::npos) {
                        response.errorCode = Timeout;
                        response.errorMessage = "Timeout was reached";
                    }
                }
                return response;
            }

            // Parse successful response
            response = parseResponse(res.body, res.statusCode);

        } catch (std::exception& e) {
            response.errorCode = UnknownError;
            response.errorMessage = e.what();
            response.status = Failed;
        }

        return response;
    }

    // Build JSON request body
    nlohmann::json buildRequestBody(const Request& request) {
        nlohmann::json body;

        body["model"] = request.model;
        if (!request.inputJson.is_null() && !request.inputJson.empty()) {
            body["input"] = request.inputJson;
        } else {
            body["input"] = request.input;
        }
        body["temperature"] = request.temperature;
        body["stream"] = false;

        if (!request.instructions.empty()) {
            body["instructions"] = request.instructions;
        }

        if (!request.conversationId.empty()) {
            body["conversation"] = request.conversationId;
        } else if (!request.previousResponseId.empty()) {
            body["previous_response_id"] = request.previousResponseId;
        }

        if (!request.metadata.empty()) {
            body["metadata"] = request.metadata;
        }

        if (!request.responseFormat.empty()) {
            body["text"]["format"] = request.responseFormat;
        }

        return body;
    }

    // Parse HTTP response
    Response parseResponse(const std::string& responseBody, int statusCode) {
        Response response;

        if (statusCode == 200) {
            try {
                auto json = nlohmann::json::parse(responseBody);
                response.fullResponse = json;

                if (json.contains("id")) {
                    response.id = json["id"].get<std::string>();
                }
                if (json.contains("model")) {
                    response.model = json["model"].get<std::string>();
                }

                // Extract status
                if (json.contains("status")) {
                    auto s = json["status"].get<std::string>();
                    if (s == "completed")    response.status = Completed;
                    else if (s == "in_progress") response.status = InProgress;
                    else if (s == "failed")  response.status = Failed;
                    else if (s == "incomplete") response.status = Incomplete;
                }

                // Extract text from output[].content[].text
                if (json.contains("output") && json["output"].is_array()) {
                    for (auto& item : json["output"]) {
                        if (item.contains("type") && item["type"] == "message") {
                            if (item.contains("content") && item["content"].is_array()) {
                                for (auto& content : item["content"]) {
                                    if (content.contains("type") && content["type"] == "output_text") {
                                        if (content.contains("text")) {
                                            response.text += content["text"].get<std::string>();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Extract usage
                if (json.contains("usage")) {
                    auto& usage = json["usage"];
                    if (usage.contains("input_tokens"))
                        response.inputTokens = usage["input_tokens"].get<int>();
                    if (usage.contains("output_tokens"))
                        response.outputTokens = usage["output_tokens"].get<int>();
                    if (usage.contains("total_tokens"))
                        response.totalTokens = usage["total_tokens"].get<int>();
                }

                response.errorCode = Success;

            } catch (std::exception& e) {
                response.errorCode = JSONParseError;
                response.errorMessage = e.what();
                response.status = Failed;
            }
        } else {
            try {
                auto json = nlohmann::json::parse(responseBody);
                response.fullResponse = json;
                response.errorCode = parseErrorCode(statusCode, json);
                response.errorMessage = getErrorMessage(response.errorCode);
                if (json.contains("error") && json["error"].contains("message")) {
                    response.errorMessage += ": " + json["error"]["message"].get<std::string>();
                }
            } catch (...) {
                response.errorCode = parseErrorCode(statusCode, nlohmann::json());
                response.errorMessage = getErrorMessage(response.errorCode);
            }
            response.status = Failed;
        }

        return response;
    }

    // Map HTTP status to error code
    ErrorCode parseErrorCode(int statusCode, const nlohmann::json& errorJson) {
        if (statusCode == 401) return InvalidAPIKey;
        if (statusCode >= 500 && statusCode < 600) return ServerError;
        if (statusCode == 429) return RateLimitExceeded;
        if (statusCode == 408) return Timeout;
        if (statusCode == 400) {
            if (!errorJson.empty() && errorJson.contains("error") && errorJson["error"].contains("type")) {
                auto t = errorJson["error"]["type"].get<std::string>();
                if (t.find("model") != std::string::npos) return InvalidModel;
                if (t.find("token") != std::string::npos) return TokenLimitExceeded;
            }
            return BadRequest;
        }
        return UnknownError;
    }

    // HTTP client
    HttpClient http_;

    // Configuration
    std::string apiKey_;
    std::string defaultModel_ = "gpt-4o";
    float defaultTemperature_ = 1.0f;
    std::string defaultInstructions_;
    int timeoutSeconds_ = 60;
    bool verbose_ = false;

    // Conversation state
    std::string currentConversationId_;
    std::string lastResponseId_;

    // Thread-safe queues
    std::deque<Request> requestQueue_;
    std::deque<Response> responseQueue_;
    std::mutex requestMutex_;
    std::mutex responseMutex_;

    // Worker thread
    std::thread workerThread_;
    std::atomic<bool> running_{false};
};

} // namespace tcx

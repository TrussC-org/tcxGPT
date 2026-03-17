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
        http_.setTimeout(120); // image generation b64_json can take a while

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

    // Image generation request
    struct ImageRequest {
        std::string prompt;
        std::string model = "gpt-image-1";
        std::string size = "1024x1024";
        std::string quality = "medium"; // low, medium, high
        std::string outputFormat = "png"; // png, jpeg, webp
        std::vector<std::string> inputImages; // base64-encoded input images (legacy)
        std::string inputImageRaw; // raw PNG/JPEG bytes for editing (preferred)
    };

    // Image generation response
    struct ImageResponse {
        bool ok = false;
        std::string error;
        std::string imageData; // Raw binary (PNG/JPEG)
    };

    // Queue an image generation request (gpt-image-1)
    void generateImage(const ImageRequest& request) {
        std::lock_guard<std::mutex> lock(imageRequestMutex_);
        imageRequestQueue_.push_back(request);
    }

    // Check if there are new image responses
    bool hasNewImageResponse() {
        std::lock_guard<std::mutex> lock(imageResponseMutex_);
        return !imageResponseQueue_.empty();
    }

    // Get the next image response
    ImageResponse getNextImageResponse() {
        std::lock_guard<std::mutex> lock(imageResponseMutex_);
        ImageResponse response;
        if (!imageResponseQueue_.empty()) {
            response = imageResponseQueue_.front();
            imageResponseQueue_.pop_front();
        }
        return response;
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
                continue;
            }

            // Check for image generation requests
            ImageRequest imageRequest;
            bool hasImageRequest = false;
            {
                std::lock_guard<std::mutex> lock(imageRequestMutex_);
                if (!imageRequestQueue_.empty()) {
                    imageRequest = imageRequestQueue_.front();
                    imageRequestQueue_.pop_front();
                    hasImageRequest = true;
                }
            }

            if (hasImageRequest) {
                ImageResponse imageResponse = processImageRequest(imageRequest);
                {
                    std::lock_guard<std::mutex> lock(imageResponseMutex_);
                    imageResponseQueue_.push_back(std::move(imageResponse));
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

    // Process image generation request (with retry)
    ImageResponse processImageRequest(const ImageRequest& request) {
        // Route to edit endpoint if input images provided
        if (!request.inputImageRaw.empty() || !request.inputImages.empty()) {
            return processImageEditRequest(request);
        }
        return processImageGenerateRequest(request);
    }

    // Generate new image (no input image)
    ImageResponse processImageGenerateRequest(const ImageRequest& request) {
        ImageResponse response;

        HttpClient dalleHttp;
        dalleHttp.setBaseUrl("https://api.openai.com");
        dalleHttp.setBearerToken(apiKey_);
        dalleHttp.setTimeout(180);
        dalleHttp.setVerbose(true);

        nlohmann::json body = {
            {"model", request.model},
            {"prompt", request.prompt},
            {"n", 1},
            {"size", request.size},
            {"quality", request.quality},
            {"output_format", request.outputFormat}
        };

        for (int attempt = 0; attempt < 3; attempt++) {
            response = {};
            try {
                if (attempt > 0) std::this_thread::sleep_for(std::chrono::seconds(2));

                auto res = dalleHttp.post("/v1/images/generations", body);
                if (!res.ok()) {
                    response.error = "Image generation error: HTTP" + std::to_string(res.statusCode);
                    try {
                        auto j = res.json();
                        if (j.contains("error") && j["error"].contains("message"))
                            response.error += " - " + j["error"]["message"].get<std::string>();
                    } catch (...) {}
                    continue;
                }

                auto json = res.json();
                if (json.contains("data") && json["data"].is_array() && !json["data"].empty()) {
                    std::string b64 = json["data"][0].value("b64_json", "");
                    if (!b64.empty()) {
                        response.imageData = decodeBase64(b64);
                        response.ok = true;
                    }
                }
                if (!response.ok && response.error.empty()) response.error = "No image data in response";
                if (response.ok) break;
            } catch (std::exception& e) {
                response.error = std::string("Image generation exception:") + e.what();
            }
        }
        return response;
    }

    // Edit image using /v1/images/edits with multipart form-data (curl)
    ImageResponse processImageEditRequest(const ImageRequest& request) {
        ImageResponse response;

#ifdef TCX_HTTP_CURL
        for (int attempt = 0; attempt < 3; attempt++) {
            response = {};
            try {
                if (attempt > 0) std::this_thread::sleep_for(std::chrono::seconds(2));

                CURL* curl = curl_easy_init();
                if (!curl) { response.error = "Failed to init curl"; continue; }

                std::string url = "https://api.openai.com/v1/images/edits";
                curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
                curl_easy_setopt(curl, CURLOPT_TIMEOUT, 180L);

                // Auth header
                struct curl_slist* headers = nullptr;
                std::string authHeader = "Authorization: Bearer " + apiKey_;
                headers = curl_slist_append(headers, authHeader.c_str());
                curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

                // Get image bytes (prefer raw, fallback to base64 decode)
                std::string imageBytes;
                if (!request.inputImageRaw.empty()) {
                    imageBytes = request.inputImageRaw;
                } else if (!request.inputImages.empty()) {
                    std::string b64 = request.inputImages[0];
                    auto commaPos = b64.find(',');
                    if (commaPos != std::string::npos) b64 = b64.substr(commaPos + 1);
                    imageBytes = decodeBase64(b64);
                }

                // Build multipart form
                curl_mime* mime = curl_mime_init(curl);

                // image file
                curl_mimepart* part = curl_mime_addpart(mime);
                curl_mime_name(part, "image");
                curl_mime_data(part, imageBytes.data(), imageBytes.size());
                curl_mime_filename(part, "input.png");
                curl_mime_type(part, "image/png");

                // prompt
                part = curl_mime_addpart(mime);
                curl_mime_name(part, "prompt");
                curl_mime_data(part, request.prompt.c_str(), request.prompt.size());

                // model
                part = curl_mime_addpart(mime);
                curl_mime_name(part, "model");
                curl_mime_data(part, request.model.c_str(), request.model.size());

                // size
                part = curl_mime_addpart(mime);
                curl_mime_name(part, "size");
                curl_mime_data(part, request.size.c_str(), request.size.size());

                // n
                part = curl_mime_addpart(mime);
                curl_mime_name(part, "n");
                curl_mime_data(part, "1", 1);

                // quality
                part = curl_mime_addpart(mime);
                curl_mime_name(part, "quality");
                curl_mime_data(part, request.quality.c_str(), request.quality.size());

                // output_format (ensure b64_json response)
                std::string outFmt = request.outputFormat.empty() ? "png" : request.outputFormat;
                part = curl_mime_addpart(mime);
                curl_mime_name(part, "output_format");
                curl_mime_data(part, outFmt.c_str(), outFmt.size());

                curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);

                // Response
                std::string responseBody;
                curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,
                    +[](void* contents, size_t size, size_t nmemb, void* userp) -> size_t {
                        auto* resp = static_cast<std::string*>(userp);
                        resp->append(static_cast<char*>(contents), size * nmemb);
                        return size * nmemb;
                    });
                curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBody);

                CURLcode curlRes = curl_easy_perform(curl);
                long httpCode = 0;
                curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);

                curl_mime_free(mime);
                curl_slist_free_all(headers);
                curl_easy_cleanup(curl);

                if (curlRes != CURLE_OK) {
                    response.error = "curl error: " + std::string(curl_easy_strerror(curlRes));
                    continue;
                }

                if (httpCode != 200) {
                    response.error = "Image edit error: HTTP" + std::to_string(httpCode);
                    try {
                        auto j = nlohmann::json::parse(responseBody);
                        if (j.contains("error") && j["error"].contains("message"))
                            response.error += " - " + j["error"]["message"].get<std::string>();
                    } catch (...) {}
                    continue;
                }

                auto json = nlohmann::json::parse(responseBody);
                if (json.contains("data") && json["data"].is_array() && !json["data"].empty()) {
                    // Try b64_json first
                    std::string b64 = json["data"][0].value("b64_json", "");
                    if (!b64.empty()) {
                        response.imageData = decodeBase64(b64);
                        response.ok = true;
                    }
                    // Fallback: download from URL
                    if (!response.ok) {
                        std::string imgUrl = json["data"][0].value("url", "");
                        if (!imgUrl.empty()) {
                            CURL* dl = curl_easy_init();
                            if (dl) {
                                std::string imgBody;
                                curl_easy_setopt(dl, CURLOPT_URL, imgUrl.c_str());
                                curl_easy_setopt(dl, CURLOPT_TIMEOUT, 60L);
                                curl_easy_setopt(dl, CURLOPT_WRITEFUNCTION,
                                    +[](void* c, size_t s, size_t n, void* u) -> size_t {
                                        static_cast<std::string*>(u)->append(static_cast<char*>(c), s*n);
                                        return s*n;
                                    });
                                curl_easy_setopt(dl, CURLOPT_WRITEDATA, &imgBody);
                                if (curl_easy_perform(dl) == CURLE_OK && !imgBody.empty()) {
                                    response.imageData = imgBody;
                                    response.ok = true;
                                }
                                curl_easy_cleanup(dl);
                            }
                        }
                    }
                }
                if (!response.ok && response.error.empty()) response.error = "No image data in response";
                if (response.ok) break;
            } catch (std::exception& e) {
                response.error = std::string("Image edit exception: ") + e.what();
            }
        }
#else
        // Fallback: try generation without input image
        ImageRequest genReq = request;
        genReq.inputImages.clear();
        return processImageGenerateRequest(genReq);
#endif
        return response;
    }

    // Base64 decode
    static std::string decodeBase64(const std::string& encoded) {
        static const std::string chars =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string decoded;
        decoded.reserve(encoded.size() * 3 / 4);
        int val = 0, bits = -8;
        for (unsigned char c : encoded) {
            if (c == '=' || c == '\n' || c == '\r') continue;
            auto pos = chars.find(c);
            if (pos == std::string::npos) continue;
            val = (val << 6) | (int)pos;
            bits += 6;
            if (bits >= 0) {
                decoded.push_back((char)((val >> bits) & 0xFF));
                bits -= 8;
            }
        }
        return decoded;
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

    // Image generation queues
    std::deque<ImageRequest> imageRequestQueue_;
    std::deque<ImageResponse> imageResponseQueue_;
    std::mutex imageRequestMutex_;
    std::mutex imageResponseMutex_;

    // Worker thread
    std::thread workerThread_;
    std::atomic<bool> running_{false};
};

} // namespace tcx

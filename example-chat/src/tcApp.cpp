#include "tcApp.h"

void tcApp::setup() {
    // Load API key from bin/data/secrets.json
    Json secrets = loadJson("secrets.json");
    string apiKey;
    if (secrets.contains("openai_api_key")) {
        apiKey = secrets["openai_api_key"].get<string>();
    }

    if (apiKey.empty() || apiKey == "YOUR_API_KEY_HERE") {
        logError() << "Please set your OpenAI API key in bin/data/secrets.json";
        messages.push_back("[Error] No API key configured");
        return;
    }

    gpt.setup(apiKey);
    gpt.setVerbose(true);
    ready = true;

    messages.push_back("[System] GPT ready. Press SPACE to send a message.");
}

void tcApp::update() {
    if (!ready) return;

    while (gpt.hasNewResponse()) {
        auto res = gpt.getNextResponse();
        if (res.errorCode == GPT::Success) {
            messages.push_back("[GPT] " + res.text);
            totalTokens += res.totalTokens;
        } else {
            messages.push_back("[Error] " + res.errorMessage);
        }
    }
}

void tcApp::draw() {
    clear(0.12f);

    setColor(1);
    drawBitmapString("tcxGPT Example - Chat", 20, 20);
    drawBitmapString("Press SPACE to send a message", 20, 40);
    drawBitmapString("Total tokens: " + to_string(totalTokens), 20, 60);

    // Draw messages
    float y = 100;
    for (auto& msg : messages) {
        // Word-wrap long messages
        string line;
        int lineLen = 0;
        for (char c : msg) {
            line += c;
            lineLen++;
            if (lineLen > 80 && c == ' ') {
                drawBitmapString(line, 20, y);
                y += 16;
                line.clear();
                lineLen = 0;
            }
        }
        if (!line.empty()) {
            drawBitmapString(line, 20, y);
            y += 16;
        }
        y += 4;
    }
}

void tcApp::keyPressed(int key) {
    if (key == ' ' && ready) {
        string msg = "Hello! Tell me a fun fact about programming.";
        messages.push_back("[You] " + msg);
        gpt.sendMessage(msg);
    }
}

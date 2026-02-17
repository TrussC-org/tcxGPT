#pragma once

#include <TrussC.h>
#include <tcxGPT.h>
using namespace std;
using namespace tc;
using namespace tcx;

class tcApp : public App {
public:
    void setup() override;
    void update() override;
    void draw() override;
    void keyPressed(int key) override;

private:
    GPT gpt;
    vector<string> messages;
    int totalTokens = 0;
    bool ready = false;
};

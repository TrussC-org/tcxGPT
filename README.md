# tcxGPT

OpenAI Responses API client addon for [TrussC](https://github.com/TrussC-org/TrussC).

Port of [ofxResponsesAPI](https://github.com/tettou771/ofxResponsesAPI) for TrussC.

## Features

- Asynchronous HTTP requests via background thread
- OpenAI Responses API (chat completions)
- Token usage tracking
- Configurable model, temperature, and system instructions
- JSON response format support

## Dependencies

- **tcxCurl** (included in TrussC addons)

## Setup

1. Add `tcxGPT` and `tcxCurl` to your project's `addons.make`:
   ```
   tcxCurl
   tcxGPT
   ```

2. Run projectGenerator to update your project.

3. Create `bin/data/secrets.json` with your OpenAI API key:
   ```json
   {
       "openai_api_key": "sk-proj-..."
   }
   ```

## Usage

```cpp
#include <TrussC.h>
#include <tcxGPT.h>
using namespace std;
using namespace tc;
using namespace tcx;

GPT gpt;

void setup() {
    gpt.setup("YOUR_API_KEY");
    gpt.setModel("gpt-4o-mini");
    gpt.setInstructions("You are a helpful assistant.");
}

void update() {
    if (gpt.hasNewResponse()) {
        auto response = gpt.getNextResponse();
        logNotice() << response.text;
        logNotice() << "Tokens: " << response.totalTokens;
    }
}

void keyPressed(int key) {
    if (key == ' ') {
        gpt.sendMessage("Hello, how are you?");
    }
}
```

### Advanced: Custom Request

```cpp
GPT::Request req;
req.input = "Translate to Japanese: Hello world";
req.model = "gpt-4o";
req.temperature = 0.3;
gpt.sendRequest(req);
```

## API Reference

### GPT

| Method | Description |
|--------|-------------|
| `setup(apiKey)` | Initialize with API key |
| `sendMessage(input)` | Send a simple message |
| `sendRequest(request)` | Send a custom request |
| `hasNewResponse()` | Check for pending responses |
| `getNextResponse()` | Get next response from queue |
| `setModel(model)` | Set model (default: `gpt-4o-mini`) |
| `setTemperature(temp)` | Set temperature (0.0-2.0) |
| `setInstructions(text)` | Set system instructions |
| `setMaxOutputTokens(n)` | Set max output tokens |
| `setResponseFormat(json)` | Set structured output format |

### GPT::Response

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Response text |
| `id` | string | Response ID |
| `status` | ResponseStatus | Completed/Failed/Incomplete |
| `errorCode` | ErrorCode | Success or error type |
| `errorMessage` | string | Error details |
| `totalTokens` | int | Total tokens used |
| `inputTokens` | int | Input tokens |
| `outputTokens` | int | Output tokens |

## Example

See `example-chat/` for a complete working example.

## License

MIT

Examples using the [Semantic Kernel](https://github.com/microsoft/semantic-kernel) in Python

# Get Started with Semantic Kernel ⚡

Install the latest package:

    python -m pip install --upgrade semantic-kernel


## OpenAI / Azure OpenAI API keys

Make sure you have an
[Open AI API Key](https://openai.com/api/) or
[Azure Open AI service key](https://learn.microsoft.com/azure/cognitive-services/openai/quickstart?pivots=rest-api)

Copy those keys into a `.env` file (see the `.env.example` file):

```
OPENAI_API_KEY=""
OPENAI_ORG_ID=""
AZURE_OPENAI_DEPLOYMENT_NAME=""
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_API_KEY=""
```

# Samples Included
- Build a question answering chatbot over a text file of Paul Graham's essays [(link)](paul-graham-essay-qa/)
- Semantically search over a CSV file with embeddings [(link)](chatbot-with-csv)


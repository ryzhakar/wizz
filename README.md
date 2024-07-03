# Wizz: AI-Enhanced Document Tool

Wizz is a proof-of-concept tool that helps you work with large collections of documents using AI. It can understand your documents and answer questions about them.

## What Wizz Does

1. Stores your documents and organizes them
2. Finds connections between different parts of your documents
3. Lets you search through your documents quickly
4. Uses AI to answer questions about your documents

## How to Use Wizz

1. Install it:
   ```
   pip install wizz
   ```

2. Set up AI access (you'll need an OpenAI account):
   ```
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

3. Use Wizz:
   ```
   wizz knowledge load --context-name "my_docs" --load-path "/path/to/your/documents"
   wizz knowledge index --context-name "my_docs"
   wizz knowledge interact --context-name "my_docs"
   ```

## Main Commands

- `load`: Add your documents to Wizz
- `index`: Prepare your documents for searching and find connections
- `search`: Look for information in your documents
- `interact`: Ask questions about your documents and get AI-powered answers
- `delete`: Remove a set of documents from Wizz

For more details, type:
```
wizz knowledge --help
```

## What You Need

- Python 3.11 or newer
- An OpenAI API key

## Technical Notes

While Wizz currently works as a standalone tool, its core features are designed with future web integration in mind.
The code uses async programming, preparing it for potential use with ASGI web servers in the future.
I developed a custom async wrapper library called `async-annoy` for the vector search tool 'annoy' (by Spotify). You can find it [here](https://github.com/ryzhakar/async-annoy).

## Important Note

Wizz is an experimental proof-of-concept and learning tool. It's not ready for real-world use. Using it with OpenAI's API may incur costs.

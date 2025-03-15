# Personal Context Server

![context-server](https://github.com/Maclean-D/context-server/raw/main/context-server.png)

A server that puts your Limitless AI lifelog into indexed markdown files, and makes it available through a REST API and web interface.

## Features

- Fetches Lifelog entries
- Persistent vector database using FAISS
- Multiple retrieval modes:
  - Day: Search across entire days
  - Memory: Search within specific memories
  - Section: Search within sections of memories
  - Line: Search individual lines/quotes
- Recency-weighted search results
- Web UI for settings and queries
- REST API for integration

## First Time Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Maclean-D/context-server.git
   cd context-server
   ```

2. Get a [Limitless API key](https://app.limitless.ai) (Account > Developers)

3. Enter it in `config.json`

3. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Start the server:
   ```bash
   python main.py
   ```

6. Continute to `Starting The Server`

## Starting The Server

1. activate a virtual environment:
   ```bash
   # Windows
   .\venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

2. Start the server:
   ```bash
   python main.py
   ```

3. 6. View server status and change settings at http://localhost:5712

## Using the API

Once your server is started visit http://localhost:5712/docs#/ to view documentation.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maclean-D/context-server&type=Date)](https://star-history.com/#Maclean-D/context-server&Date)

## Contributors

<a href="https://github.com/Maclean-D/context-server/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Maclean-D/context-server" />
</a>

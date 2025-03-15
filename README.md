# Personal Context Server

![context-server](https://github.com/Maclean-D/personal-context/raw/main/context-server.png)

A powerful server that indexes and retrieves your personal context from Markdown files, making it available through a REST API and web interface.

## Features

- Continuous monitoring and indexing of Markdown files
- Multiple retrieval modes:
  - Day: Search across entire days
  - Memory: Search within specific memories
  - Section: Search within sections of memories
  - Line: Search individual lines/quotes
- Automatic timezone conversion
- Recency-weighted search results
- Web UI for settings and queries
- REST API for integration
- Persistent vector database using FAISS

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Maclean-D/context-server.git
   cd context-server
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the server:
   ```bash
   python main.py
   ```

The server will start on `http://localhost:5712` by default.

## Usage

1. **File Organization**
   - Place your Markdown files in the `notes/YEAR/MONTH/` directory structure
   - Files are automatically indexed when added or modified

2. **Web Interface**
   - Access `http://localhost:5712` in your browser
   - Configure settings
   - Test queries
   - Monitor indexing progress

3. **API Integration**
   - Works, no docs yet WIP.

## File Format

Your Markdown files should follow this structure:

```markdown
# Memory Title
Regular text under the memory

## Section Title
Section content

> Quote or specific line to remember
More section content
```

## Configuration

Default settings in `config.json`:
```json
{
  "docs_dir": "./notes",
  "timezone": "auto",
  "include_titles": true,
  "retrieval_mode": "memory",
  "recency_weight": 0.0,
  "n_candidates": 10,
  "n_results": 5,
  "port": 5712
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Maclean-D/context-server&type=Date)](https://star-history.com/#Maclean-D/context-server&Date)

## Contributors

<a href="https://github.com/Maclean-D/context-server/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Maclean-D/context-server" />
</a>
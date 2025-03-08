import re
from datetime import datetime
import os
from .config import config, TZ

def normalize_timestamps(text: str) -> str:
    """Convert millisecond timestamps to human-readable format and remove end time."""
    # First remove endMs parameter
    text = re.sub(r'&endMs=\d{13}', '', text)
    
    # Then convert startMs to human-readable format
    pattern = re.compile(r'#?startMs\s*[:=]\s*(\d{13})')
    def _replace(match):
        ms = int(match.group(1))
        dt = datetime.fromtimestamp(ms/1000.0, TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return pattern.sub(_replace, text)

def get_file_date(path: str):
    """Extract date from file path or name."""
    fname = os.path.splitext(os.path.basename(path))[0]
    parts = fname.replace('_', '-').split('-')
    year = month = day = None
    try:
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
    except:
        try:
            year = int(os.path.basename(os.path.dirname(os.path.dirname(path))))
            month = int(os.path.basename(os.path.dirname(path)))
            day = int(parts[0])
        except:
            return None
    try:
        return datetime(year, month, day)
    except:
        return None

def parse_markdown_content(content: str):
    """Parse markdown content into different segment types."""
    segments = {'day': [], 'memory': [], 'section': [], 'line': []}
    lines = content.splitlines()
    
    # Parse whole day content
    day_lines = []
    for line in lines:
        if line.startswith('# '):
            continue  # Skip memory titles in day content
        elif line.startswith('## '):
            day_lines.append(line[3:].strip())
        elif line.startswith('>'):
            day_lines.append(line[1:].strip())
        else:
            day_lines.append(line)
    day_text = "\n".join(day_lines).strip()
    if day_text:
        segments['day'].append({"text": day_text, "title": None})

    # Parse memories, sections, and lines
    current_mem_title = None
    current_mem_lines = []
    current_mem_all_lines = []  # All lines in memory including sections
    current_sec_title = None
    current_sec_lines = []
    current_sec_all_lines = []  # All lines in section including subsections

    for line in lines:
        if line.startswith('# '):
            # Close previous memory if exists
            if current_mem_title is not None:
                if current_sec_title is not None:
                    sec_text = "\n".join(current_sec_lines).strip()
                    sec_all_text = "\n".join(current_sec_all_lines).strip()
                    if sec_all_text:
                        text = f"## {current_sec_title}\n{sec_all_text}" if config["include_titles"] else sec_all_text
                        segments['section'].append({
                            "text": text,
                            "title": current_sec_title,
                            "file_memory": current_mem_title
                        })
                # Add memory with all content
                mem_all_text = "\n".join(current_mem_all_lines).strip()
                if mem_all_text:
                    text = f"# {current_mem_title}\n{mem_all_text}" if config["include_titles"] else mem_all_text
                    segments['memory'].append({
                        "text": text,
                        "title": current_mem_title
                    })
            
            # Start new memory
            current_mem_title = line[2:].strip()
            current_mem_lines = []
            current_mem_all_lines = []  # Don't include title in all_lines
            current_sec_title = None
            current_sec_lines = []
            current_sec_all_lines = []

        elif line.startswith('## '):
            # Close previous section if exists
            if current_sec_title is not None:
                sec_text = "\n".join(current_sec_lines).strip()
                sec_all_text = "\n".join(current_sec_all_lines).strip()
                if sec_all_text:
                    text = f"## {current_sec_title}\n{sec_all_text}" if config["include_titles"] else sec_all_text
                    segments['section'].append({
                        "text": text,
                        "title": current_sec_title,
                        "file_memory": current_mem_title
                    })
            
            # Start new section
            current_sec_title = line[3:].strip()
            current_sec_lines = []
            current_sec_all_lines = []  # Don't include title in all_lines
            if current_mem_title is not None:
                current_mem_all_lines.append("")  # Add blank line before section
                current_mem_all_lines.append(line)  # Add section header with ## to memory

        elif line.startswith('>'):
            line_text = line[1:].strip()
            if line_text:
                segments['line'].append({
                    "text": line_text,
                    "title": None,
                    "file_memory": current_mem_title,
                    "file_section": current_sec_title
                })
            if current_sec_title is not None:
                current_sec_lines.append(line_text)
                current_sec_all_lines.append(line)  # Add to section with quote
                current_mem_all_lines.append(line)  # Add to memory with quote
            elif current_mem_title is not None:
                current_mem_lines.append(line_text)
                current_mem_all_lines.append(line)  # Add to memory with quote
            continue

        if current_sec_title is not None:
            if not line.startswith('## '):
                if not line.startswith('>'):
                    current_sec_lines.append(line)
                    current_sec_all_lines.append(line)  # Add to section
                    current_mem_all_lines.append(line)  # Add to memory
        elif current_mem_title is not None:
            if not line.startswith('# '):
                if not line.startswith('>'):
                    current_mem_lines.append(line)
                    current_mem_all_lines.append(line)  # Add to memory

    # Close final memory/section if exists
    if current_mem_title is not None:
        if current_sec_title is not None:
            sec_text = "\n".join(current_sec_lines).strip()
            sec_all_text = "\n".join(current_sec_all_lines).strip()
            if sec_all_text:
                text = f"## {current_sec_title}\n{sec_all_text}" if config["include_titles"] else sec_all_text
                segments['section'].append({
                    "text": text,
                    "title": current_sec_title,
                    "file_memory": current_mem_title
                })
        # Add final memory with all content
        mem_all_text = "\n".join(current_mem_all_lines).strip()
        if mem_all_text:
            text = f"# {current_mem_title}\n{mem_all_text}" if config["include_titles"] else mem_all_text
            segments['memory'].append({
                "text": text,
                "title": current_mem_title
            })

    return segments 
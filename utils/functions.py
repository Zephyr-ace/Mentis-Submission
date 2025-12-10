def loadText(pfad: str = "data/diary.txt", start: int = None , end: int = None) -> str:
    with open(pfad, "r", encoding="utf-8") as datei:
        zeilen = datei.readlines()[start:end]
        return "".join(zeilen)

def split_by_keyword(text, keyword):
    return [e.strip() for e in text.split(keyword) if e.strip()]

def chunk_diary_entries(text: str, timeframe_hours: int = 1):
    import re
    from datetime import datetime, timedelta
    
    diary_pattern = r'Datum:\s*"([^"]+)"'
    entries = re.split(diary_pattern, text)[1:]
    
    chunks = []
    current_chunk = ""
    last_timestamp = None
    
    for i in range(0, len(entries), 2):
        if i + 1 >= len(entries):
            break
            
        timestamp_str = entries[i]
        content = entries[i + 1]
        
        try:
            timestamp = datetime.strptime(timestamp_str.split('[')[0].strip(), "%d. %B %Y um %H:%M:%S GMT%z")
        except ValueError:
            try:
                timestamp = datetime.strptime(timestamp_str.split('[')[0].strip(), "%d. %B %Y um %H:%M:%S GMT-%H")
            except ValueError:
                timestamp = None
        
        full_entry = f'Datum: "{timestamp_str}"{content}'
        
        if last_timestamp is None or timestamp is None or timestamp - last_timestamp > timedelta(hours=timeframe_hours):
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = full_entry
        else:
            current_chunk += full_entry
        
        last_timestamp = timestamp
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def load_queries(file_path: str = "evaluation/queries.json") -> list:
    import json
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        return data["queries"]

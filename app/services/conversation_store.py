#this file stores the conversation in a JSON file, read and write messanges
import os
import json
import uuid




class JsonConversationSore:
    def __init__(self, path: str = "services/conversations/conversations.json"):        #initialize empty file
        self.path = path
        if not os.path.exists(self.path):
            with open(self.path, "w" , encoding="utf-8") as f:
                json.dump({}, f)


    def _read_data(self):                                                               #reads the file
        try:
            with open (self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {} #return empty dictionary in case we have an error, so we don't crash
        

    def _write_data(self, data):                                                        #writes the file
        with open (self.path, "w", encoding="utf-8") as f:      #keeps unicode characters readable
            json.dump(data,f, ensure_ascii=False, indent=2)

    
    def check_conv (self, conversation_id):
        data = self._read_data()

        if conversation_id is None:
            conversation_id = str(uuid.uuid4())                 #make the id a universally unique identifier

        if conversation_id not in data:
            data[conversation_id] = {
                "conversation_id": conversation_id,
                "messages": []
            }
            self._write_data(data)
        return conversation_id

    def add_message(self, conversation_id, role, content):      #append a message to a conversation
        data = self._read_data()

        if conversation_id not in data:
            # auto-create if missing
            self.check_conv(conversation_id)
            data = self._read_data()

        data[conversation_id]["messages"].append({
            "role": role,
            "content": content
        })

        self._write_data(data)


    def get_messages(self, conversation_id):                    #retrieve full conversation
        data = self._read_data()
        return data.get(conversation_id, {}).get("messages", [])


    
    

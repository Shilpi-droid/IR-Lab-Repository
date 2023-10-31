
indexMapping = {
    "type_name" : {
    "properties" : {
    "settings": {
        "index": {
            "similarity": {
                "my_bm25": {
                    "type": "BM25",
                    "b": 0.75,  # You can adjust these parameters as needed
                    "k1": 1.2
                }
            }
        }
    }}},
    "mappings": {
         
        "properties": {
            "did": {
                "type": "integer"
            },
            "text": {
                "type": "text",
                "similarity": "my_bm25"  # Apply the BM25 similarity to the "text" field
            },
            "TextDescVec": {
                "type": "dense_vector",
                "dims": 768
            }
        }}
    }


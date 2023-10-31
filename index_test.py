indexMapping = {
  "mappings": {
    "properties": {
      "TextDescVec": {
        "type": "dense_vector",
        "dims": 768
      },
      "text": {
        "type": "text",
        "similarity": "my_bm25"
      },
      "did": {
        "type": "integer"
      }
    }
  },
  "settings": {
    "index": {
      "similarity": {
        "my_bm25": {
          "type": "BM25",
          "b": 0.75,
          "k1": 1.2
        }
      }
    }
  }
}


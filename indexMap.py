indexMapping = {
    "properties":{
        "did":{
            "type":"long"
        },
        "text":{
            "type":"text"
        },
        "TextDescVec":{
            "type":"dense_vector",
            "dims":768,
            "index":True,
            "similarity": "cosine",
        }
    }
}
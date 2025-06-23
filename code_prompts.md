# Drafting initial code files with LLMs

## Aim: Extract survivors' journey (or movement) details from their testimony transcripts using an LLM

### 0. The input and output specification are as follows:
    - input: a holocaust survivor transcript file (.txt) or a folder containing multiple transcript (.txt) files or a compressed folder of transcripts in any format (.zip, .rar, .7z, .tar, .tar.gz or .tgz, .xz, .gz, .bz2, .zst)
    - output: a jsonl file with chronologically arranged jsonlines each containing the list of journeys.
        -Example journey detail: 
            {"from_location": "Budapest, Hungary", 
            "to_location": "Belgrade, Yugoslavia", 
            "approx_date": "October 1956", 
            "mode_of_transport": "Not mentioned", 
            "reason": "Fled Hungary during the Hungarian Revolution", 
            "context": "Then when the Hungarian revolution just erupted. And the children were, by the divorce process, given to me anyway. So I grabbed the two children. I went out, and I left the country, went to Yugoslavia.",
            "transcript": "10162.txt",
            "source_lat": 47.4978789, "source_lon": 19.0402383, 
            "target_lat": 44.7880163, "target_lon": 20.4522489
            }

### 1. Create a well crafted/engineered prompt to extract and return unique journey/movement details in the above format, including a short context from the text that serves as an evidence that movement happened


### 2. For the geocoding (i.e. getting the place coordinates) the places mentioned, we adopt the following approach:
     for each journey entry, we use a combination of the gazetteer and LLM in this way:
        - read the original gazetteer file into memory
        - for each location in the set of all unique locations
            - if the location entry exists in the gazetter, update the journey entry
            - else, use the LLM to estimate the location coordinates based on the context. Update the journey entry with lat lon details and update the gazetteer list in memory
            - compare the original and the updated verion of the gazetteer and save the difference for manual validation

### 3. Optimize for CLI based, multiprocessing support with user friendly progress monitoring and messaging

### 4. Guard against hitting a token size limit by chunking, recombininhg and deduplication

### 5. Wrap the process up with the appropriate message about where everything is.

### 6. Make it production ready with an appropriate level of documentation 

Let me know what you think and feel free to suggest any interesting feature.
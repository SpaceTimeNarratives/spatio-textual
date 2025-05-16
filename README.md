# spatio-textual
A library for spatial textual analysis with a focus on the Corpus of Lake District Writing and Holocaust survivors' testimonies.

1. Clone this repo
```bash
git clone https://github.com/SpaceTimeNarratives/spatio-textual.git
```
2. Run `setup.sh` to
  - Create a venv/ folder
  - Activate it 
  - Install everything from requirements.txt
  - Download the defaul model, spacy's `en-core-web-trf` 

```bash
./setup.sh
```
3. Activate virtual environment
```bash
source venv/bin/activate
```

4. Run `entity annotator.py` on the segments
```bash
python entity_annotator.py -i segments.jsonl -o out_dir -r resources -w 4
```

# wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# wget http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv
# unzip annotations_trainval2017.zip

# mkdir ../.data
# mv quora_duplicate_questions.tsv ../.data/quora_duplicate_questions.tsv
# mv annotations/ ../.data/annotations/
# rm annotations_trainval2017.zip

wget http://cs.jhu.edu/~vandurme/data/parabank-1.0-5m-diverse.zip
unzip parabank-1.0-5m-diverse.zip
mv parabank-1.0-small-diverse/ ../.data/parabank-1.0-small-diverse/
rm parabank-1.0-5m-diverse.zip


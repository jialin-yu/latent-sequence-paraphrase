wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv
unzip annotations_trainval2014.zip

mkdir ../.data
mv quora_duplicate_questions.tsv ../.data/quora_duplicate_questions.tsv
mv annotations/ ../.data/annotations/
rm annotations_trainval2014.zip
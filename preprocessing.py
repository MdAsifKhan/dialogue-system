import json
import pandas as pd
from torchtext import data
import os
import pdb
from torchtext.vocab import GloVe

# Convert Dialog to Sequence and kb to triple format
def create_data_csv(json_file_name='data/kvret_train_public.json'):
	data = json.load(open(json_file_name))	
	#Save Dialog as a csv file
	speaker_1 = []
	speaker_1_triples = []
	speaker_2 = []
	speaker_2_triples = []
	eou_token = 'eofu'
	bou_token = 'bofu'
	triples = []
	for i,dialog in enumerate(data):
		driver = ''
		assistant = ''
		if not dialog['dialogue']:
			speaker_1.append('')
			speaker_2.append('')		
			continue
		for turn in dialog['dialogue']:
			turn_data = turn['data']
			turn_ = turn['turn']
			if turn_ == 'assistant':
				assistant = assistant + ' ' + bou_token+ ' ' + turn_data['utterance'] + ' ' + eou_token
			else:
				driver = driver + ' ' + bou_token+ ' ' + turn_data['utterance'] + ' ' + eou_token
		if data[i]['dialogue'][0]['turn'] == 'assistant':
			speaker_1.append(assistant)
			speaker_2.append(driver)
		else:
			speaker_1.append(driver)
			speaker_2.append(assistant)		

		kb_table = dialog['scenario']['kb']
		kb_column = kb_table['column_names']
		sub_ = kb_column[0]
		if not kb_table['items']:
			triples.append([i,kb_table['kb_title'],'','',''])
			continue
		for row in kb_table['items']:
			for key,value in row.items():
				if key == sub_:
					continue
				triples.append([i, kb_table['kb_title'], row[sub_], key, value])
	df = pd.DataFrame(list(zip(speaker_1, speaker_2)), columns=['speaker1','speaker2'])
	triples_df = pd.DataFrame(triples, columns=['sample_idx','kb_title','subject','relation','object'])
	return df, triples_df


# Save Dialogue data as a sequence
if not os.path.exists('data/train_data.csv'):
	df, triples = create_data_csv('data/kvret_train_public.json')
	df.to_csv('data/train_data.csv', index=False)	
	triples.to_csv('data/train_triples.csv', index=False)

if not os.path.exists('data/valid_data.csv'):
	df, triples = create_data_csv('data/kvret_dev_public.json')
	df.to_csv('data/valid_data.csv', index=False)
	triples.to_csv('data/valid_triples.csv', index=False)

if not os.path.exists('data/test_data.csv'):
	df, triples = create_data_csv('data/kvret_test_public.json')
	df.to_csv('data/test_data.csv', index=False)
	triples.to_csv('data/test_triples.csv', index=False)


class ProcessData:

	def __init__(self, path, train_file, val_file, test_file, max_len=None, batch_size=100, max_vocab=999999):
		self.path = path
		self.train_file = train_file
		self.val_file = val_file
		self.test_file = test_file
		self.max_len = max_len
		self.batch_size = batch_size
		self.max_vocab = max_vocab

		self.text_field = data.Field(lower=True,
								tokenize=data.get_tokenizer('spacy'),
								pad_token='<pad>',
								unk_token='<unk>',
								fix_length=self.max_len,
								include_lengths=True,
								batch_first=True)

		self.trainData, self.validData, self.testData = data.TabularDataset.splits(
								path=self.path, train=self.train_file,
								validation=self.val_file, test=self.test_file,
								format='csv',skip_header=True,
								fields=[('input', self.text_field), 
										('target', self.text_field)])

		self.text_field.build_vocab(self.trainData, max_size=self.max_vocab, 
										min_freq=1, vectors=GloVe('6B', dim=100))

		self.vocab = {'text_vocab':self.text_field.vocab.stoi,
				'text_inv_vocab':self.text_field.vocab.itos}
		self.vectors = self.text_field.vocab.vectors

	def batch_generator(self):
		train_iter, valid_iter, test_iter = data.BucketIterator.splits(
						(self.trainData, self.validData, self.testData), 
						batch_size=self.batch_size, 
						sort=False, shuffle=False,
						sort_key=lambda x: len(x.input), 
						repeat=False, sort_within_batch=False)

		return train_iter, valid_iter, test_iter 

path = 'data'
train_file = 'train_data.csv'
val_file = 'valid_data.csv'
test_file = 'test_data.csv'
dataset = ProcessData(path, train_file, val_file, test_file, max_len=100,batch_size=100,max_vocab=999999)

train_iter, valid_iter, test_iter = dataset.batch_generator()
for tr in train_iter:
	pdb.set_trace()
	tr.input
	tr.target

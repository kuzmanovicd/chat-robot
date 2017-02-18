import codecs
import string
import unicodedata
import re
import math

_Q_NAME = "ime korisnika:"
_A_NAME = "dejan kuzmanovic:"

PARSED_FILE = "parsed.txt"

EN_WHITELIST = "0123456789abcdefghijklmnopqrstuvwxyz,.:;/<>?!()+-* "
WORD_SPLITER = ",.<>?!+-"

def read_lines(filename):
	return codecs.open(filename, encoding='utf8').read().lower().strip().split('\n')[:-1]


def filter_line(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])

def strip_accents(text):
    return ''.join(char for char in
                   unicodedata.normalize('NFKD', text)
                   if unicodedata.category(char) != 'Mn')

def split_punctuation():
	if len(questions) != len(answers):
		print('Length of questions and answers is not the same...')
		return

	for i in range(len(questions)):
		for ch in WORD_SPLITER:
			questions[i] = questions[i].replace(ch, ' ' + ch)
			answers[i] = answers[i].replace(ch, ' ' + ch)

def haha():
	if len(questions) != len(answers):
		print('Length of questions and answers is not the same...')
		return

	for i in range(len(questions)):
		sent_q = []
		sent_a = []
		for word in questions[i].split(' '):
			if 'haha' in word or 'ahah' in word:
				word = 'haha'
			sent_q.append(word)

		for word in answers[i].split(' '):
			if 'haha' in word or 'ahah' in word:
				word = 'haha'
			sent_a.append(word)
		
		questions[i] = ' '.join(sent_q)
		answers[i] = ' '.join(sent_a)


def remove_http_links():
	if len(questions) != len(answers):
		return

	reg = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)')
	for i in range(len(questions)):
		if reg.match(questions[i]):
			questions.pop(i)
			answers.pop(i)
			remove_http_links()
			break

lines = read_lines('danijela.txt')
lines = [ strip_accents(line) for line in lines ]
lines = [ filter_line(line, EN_WHITELIST) for line in lines ]

alt = True
question_done = False
answer_done = False

questions = []
answers = []

def parse_lines(text):
	q_progress = False
	a_progress = False

	for line in text:

		if (math.fabs(len(answers)) - math.fabs(len(questions))) > 1:
			print('BELAJ', len(answers), len(questions))

		if line.startswith(_Q_NAME):
			a_progress = False
			str = line.replace(_Q_NAME, '').strip()
			if q_progress:
				questions[-1] += ' ' + str
			else:
				questions.append(str)
				q_progress = True
		elif line.startswith(_A_NAME):
			q_progress = False
			str = line.replace(_A_NAME, '').strip()
			if a_progress:
				answers[-1] += ' ' + str
			else:
				answers.append(str)
				a_progress = True
		else:
			str = str.strip()
			if q_progress:
				questions[-1] += ' ' + str
			elif a_progress:
				answers[-1] += ' ' + str
			else:
				print('Error')

def remove_empty_lines():
	for i in range(len(questions) - 1):
		if not questions[i].strip() or not answers[i].strip():
			questions.pop(i)
			answers.pop(i)
			remove_empty_lines()
			break


def save():
	file = open(PARSED_FILE, 'w')

	for i in range(len(answers)):
		questions[i] = questions[i].strip()
		answers[i] = answers[i].strip()

		file.write(answers[i] + '\n')
		file.write(questions[i] + '\n')

parse_lines(lines)

print(len(answers), len(questions))
remove_empty_lines()
remove_http_links()
split_punctuation()
haha()

print(len(answers), len(questions))


	#file.write(answers[i] + '\n')

	#print(_Q, i, questions[i])
	#print(_A, i, answers[i])




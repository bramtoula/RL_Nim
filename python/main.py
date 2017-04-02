import os
from time import sleep
from random import randint

def defineBoard():
	os.system('clear')
	print "Let's start by defining our game:"
	heaps = raw_input("Enter number of heaps you want: ")
	for x in range(1,int(heaps)+1):
		num = raw_input("Enter number of matches on heap %d: " % x)
		heap.append(int(num))
	print


def printBoard(heap):
	os.system('clear')
	num = 0
	for num,row in enumerate(heap):
		print num+1,
		for match in range(0,row):
			print " |",
		print

def nimSum():
	return reduce(lambda x,y: x^y, heap)

def winingHeap():
	return [x^nimSum() < x for x in heap].index(True)

def userMove():
	row, num = raw_input("Enter row and num of matches you want to take separated with space ex.(1 2):  ").split()
	row, num = int(row)-1,int(num)

	try:
		if row <= -1: raise
		if num>0 and num<=heap[row]:
			heap[row]-=num
			printBoard(heap)
		else:
			printBoard(heap)
			print "WRONG NUMBER TRY AGAIN"
			userMove()
	except:
		printBoard(heap)
		print "WRONG ROW TRY AGAIN"
		userMove()
	if isItEnd(): print "YOU WIN"


def computerMove():
	print "Now it's my turn"
	sleep(1.3)
	if nimSum()==0:
		heap[heap.index(max(heap))]-=randint(1,max(heap))
	else: heap[winingHeap()]^=nimSum()
	printBoard(heap)
	if isItEnd(): print "YOU LOST"


def isItEnd():
	return all(z == 0 for z in heap)

heap = []

os.system('clear')

print """
  _   _ _____ __  __        _____          __  __ ______
 | \ | |_   _|  \/  |      / ____|   /\   |  \/  |  ____|
 |  \| | | | | \  / |     | |  __   /  \  | \  / | |__
 | . ` | | | | |\/| |     | | |_ | / /\ \ | |\/| |  __|
 | |\  |_| |_| |  | |     | |__| |/ ____ \| |  | | |____
 |_| \_|_____|_|  |_|      \_____/_/    \_|_|  |_|______|

                                                         """
sleep(2)
os.system('clear')
print """Nim is a mathematical game of strategy in which two players
take turns removing objects from distinct heaps. On each turn, a player
must remove at least one object, and may remove any number of objects
provided they all come from the same heap. The one who can't make a move loses.
source: wikipedia.com

"""

raw_input("Press Enter to continue...")

defineBoard()

printBoard(heap)

while True:
	userMove()
	if isItEnd(): break
	computerMove()
	if isItEnd(): break

#1. 建立class
#2. 建立new node和其parameter
#3. 第一個new node, head=new_node, 並指向None, ptr=head
#   接下來的new node, ptr.next = new_node, new_node.next=None, ptr=ptr.next

class student:
	def __init__(self):
		self.name = ''
		self.no = ''
		self.math = 0
		self.next = None

init = True
select = 0

while select != 2:

	try:
		select = int(input('Insert new name, input 1 or leave, input 2:'))
	except ValueError:
		print('Value error!! Input again')

	if select == 1:
		#make new_data
		new_data = student()
		new_data.name = input('Please insert name:')
		new_data.no = input('Please insert student number:')
		new_data.math = eval(input('Please insert the math score:'))
		
		if init == True:
			head = new_data
			head.next = None
			ptr = head
			init = False
		else:
			ptr.next = new_data
			new_data.next = None
			ptr = ptr.next

ptr = head
print()
while ptr != None:
	print('name:%s\tnumber:%s\tmath score:%d' \
	      % (ptr.name,ptr.no,ptr.math))
	ptr = ptr.next

################################
def findnode(head,no):
	ptr = head
	while ptr != None:
		if ptr.no == no:
			return ptr
		ptr = ptr.next
	return ptr  #找不到則返回None

def insertnode(head, ptr, new_node):
	if ptr == None:
		#insert at first
		new_node.next = head
		head = new_node
		return new_node
	else:
		#insert at last
		if ptr.next == None:
			ptr.next = new_node
		#insert at ptr back
		else:
			new_node.next = ptr.next
			ptr.next = new_node
	return head

new_data = student()
new_data.name = 'Jack'
new_data.no = '2'
new_data.math = 97
ptr = findnode(head, new_data.no)
head = insertnode(head, ptr, new_data)

ptr = head
print()
while ptr != None:
	print('name:%s\tnumber:%s\tmath score:%d' \
	      % (ptr.name,ptr.no,ptr.math))
	ptr = ptr.next







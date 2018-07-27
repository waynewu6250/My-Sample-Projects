class example:
	pass


def function1(self, hello):
	self.hello = hello

example.__function1 = classmethod(function1)


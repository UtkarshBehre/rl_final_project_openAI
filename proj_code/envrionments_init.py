
class Envrionments_Init:

	def __init__(self, env_names=None):
		# TODO: CODE NEEDS TO BE CHANGED
		if env_names:
			self.env_list = self.make_env_list(env_names)

	def get_environments_list_to_test(self, include_image_based=True, include_non_image_based=True):
		# TODO: implementation

		# TODO: have 1 list for image based and 1 list for non image based and return them based on whether merge required or not

		# TODO: should return a list of environments
		return self.make_env("CartPole") # this needs change

	def make_env_list(self, env_names):
		env_list = []
		for env_name in env_names:
			env_list.append(self.make_env(env_name))

	def make_env(self, env_name):
		# TODO: have the image dectection code here
		image_based = True
		return Environment(env_name, image_based=image_based)
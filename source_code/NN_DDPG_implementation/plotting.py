import matplotlib

import matplotlib.pyplot as plt
import torch


def plot_durations(episode_durations):
	# set up matplotlib
	is_ipython = 'inline' in matplotlib.get_backend()
	if is_ipython:
		from IPython import display
		
	plt.figure(2)
	plt.clf()
	durations_t = torch.tensor(episode_durations, dtype=torch.float)
	plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Duration')
	plt.plot(durations_t.numpy())
	# Take 100 episode averages and plot them too
	if len(durations_t) >= 100:
		means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
		means = torch.cat((torch.zeros(99), means))
		plt.plot(means.numpy())
	
	plt.pause(0.001)  # pause a bit so that plots are updated
	if is_ipython:
		display.clear_output(wait=True)
		display.display(plt.gcf())


def plot(array_to_plot, figure_n, title, y_label):
	# set up matplotlib
	is_ipython = 'inline' in matplotlib.get_backend()
	if is_ipython:
		from IPython import display
	
	plt.figure(figure_n)
	plt.clf()
	durations_t = torch.tensor(array_to_plot, dtype=torch.float)
	plt.title(title)
	plt.xlabel('Episode')
	plt.ylabel(y_label)
	plt.plot(durations_t.numpy())
	# Take 100 episode averages and plot them too
	if len(durations_t) >= 100:
		means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
		means = torch.cat((torch.zeros(99), means))
		plt.plot(means.numpy())
	
	plt.pause(0.001)  # pause a bit so that plots are updated
	if is_ipython:
		display.clear_output(wait=True)
		display.display(plt.gcf())
# Takes dataframe and returns a numpy array that can be used for the create_plotly_ff_heatmap
def create_heatmap_array(dataframe):
	array = dataframe.pivot('y', 'x', 'temp').values

	return array


# Takes dataframe of difference in temp and returns a numpy array that can be used for create_plotly_ff_heatmap_diff
def create_heatmap_diff_array(dataframe):
	array = dataframe.pivot('y', 'x', 'diff').values

	return array


# Takes dataframe as an input and returns x and y lists for
# the axis labels that can be used for the plotly.figure_factory.create_annotated_heatmap
def create_axis_lists(dataframe):
	x_list = dataframe['x'].to_list()
	x_list = list(dict.fromkeys(x_list))
	y_list = dataframe['y'].to_list()
	y_list = list(dict.fromkeys(y_list))
	# y_list.reverse()

	return x_list, y_list


# Takes a dataframe as an input and returns a plotly.graphical_objects heatmap
def create_plotly_go_heatmap(dataframe):
	heatmap = go.Figure(go.Heatmap(
		x=dataframe['x'],
		y=dataframe['y'],
		z=dataframe['temp']))

	return heatmap


# Takes a dataframe as an input and returns a plotly.figure_factory annotated heatmap
def create_plotly_ff_heatmap_abs(dataframe):
	# colorscale_heatmap = [[0, 'rgb(0,67,206)'], [0.5, 'rgb(105,41,196)'],[1, 'rgb(162,25,31)']]
	colorscale_heatmap = [[0, 'rgb(69,137,255)'], [1, 'rgb(250,77,86)']]

	x_list, y_list = create_axis_lists(dataframe)
	heatmap_array = create_heatmap_array(dataframe)
	ts = get_timestamp(dataframe)

	heatmap = ff.create_annotated_heatmap(
		x=x_list,
		y=y_list,
		z=heatmap_array,
		colorscale=colorscale_heatmap,
		xgap=10,
		ygap=1,
		showscale=True
	)

	heatmap.update_layout(
		title=('Absolute temperature at ' + ts),
		xaxis=dict(title='Column', color='black', side='bottom'),
		yaxis=dict(title='Row', autorange="reversed"),
		plot_bgcolor='rgba(0,0,0,0)',
	)

	heatmap.data[0].colorbar = dict(title='Temperature', titleside='right')

	return heatmap


# Takes a dataframe with difference in temps as an input and returns a plotly.figure_factory annotated heatmap
def create_plotly_ff_heatmap_diff(dataframe):
	# colorscale_heatmap = [[0, 'rgb(0,67,206)'], [0.5, 'rgb(105,41,196)'],[1, 'rgb(162,25,31)']]
	max, min = dataframe['diff'].max(), dataframe['diff'].min()
	midpoint = abs(min) / (abs(max) + abs(min))
	colorscale_heatmap = [[0, 'rgb(69,137,255)'], [midpoint, 'rgb(255,255,255)'], [1, 'rgb(250,77,86)']]

	x_list, y_list = create_axis_lists(dataframe)
	heatmap_array = create_heatmap_diff_array(dataframe)
	ts0, ts1 = dataframe.columns[1], dataframe.columns[2]

	heatmap = ff.create_annotated_heatmap(
		x=x_list,
		y=y_list,
		z=heatmap_array,
		autocolorscale=False,
		colorscale=colorscale_heatmap,
		font_colors=['black'],
		xgap=10,
		ygap=1,
		showscale=True
	)

	heatmap.update_layout(
		title=('Changes in temperature between ' + ts0 + ' and ' + ts1),
		xaxis=dict(title='Column', side='bottom'),
		yaxis=dict(title='Row', autorange="reversed"),
		plot_bgcolor='rgba(0,0,0,0)',
	)

	heatmap.data[0].colorbar = dict(title='Temperature Change', titleside='right')

	return heatmap
# import needed packages
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from datetime import datetime as dt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# import flask

# formatting
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#server = flask.Flask(__name__)
#app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)



app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# colorscale1 = [[1, 'gold'], [0, 'mediumturquoise']]
# colorscale2 = [[0, 'mediumturquoise'], [1, 'lightsalmon']]

# app layout
app.layout = html.Div([
    html.H1(children='Winterthur Interface', style={'textAlign': 'center'}),
# 1. User uploads a map file and data file(s) - many data files to come later, for now only definitely works with one data file
    # button to upload "map" file
    #dcc.Upload(
        #html.Button('Upload Map File(s)'),
        #id='upload-map', style={'display': 'inline-block'},
        ## Allow multiple files to be uploaded
        #multiple=True
    #),
    ## output selected "map" filename
    #html.Div(id='output-image-upload', style={'display': 'inline-block'}),

    # button to upload "data" file
    html.Div([
        html.H5(children='Step 1: Upload data file(s) (.pm2 or .txt files supported):'),
        dcc.Upload(
            html.Button('Upload Data File(s)'),
            id='upload-data', style={'display': 'inline-block'},
            # Allow multiple files to be uploaded
            multiple=True
        ),
        # output selected data file filename
        html.Div(id='output-data-upload', style={'display': 'inline-block'}),
    ]),

# 2. User enters start date and end date for the overall period they wish to examine
    # LATER ON maybe have the dates be automatically selected by the program based on the date file entered? would be
    # difficult to scale with many data files, but pandas doesn't seem to have an issue with weird dates at least
    html.Div([html.H5('Step 2: Select overall date range to be analyzed:')]),

    # START DATE selector
    html.Div([
        html.H6('Select start date (YEAR, MONTH, DAY, HOUR, MINUTE):'),
        dcc.Input(id='startDateYear', value=2013, type='number'),
        dcc.Dropdown(id='startDateMonth',
                     options=[{'label': 'January', 'value': 1}, {'label': 'February', 'value': 2},
                              {'label': 'March', 'value': 3}, {'label': 'April', 'value': 4},
                              {'label': 'May', 'value': 5}, {'label': 'June', 'value': 6},
                              {'label': 'July', 'value': 7}, {'label': 'August', 'value': 8},
                              {'label': 'September', 'value': 9}, {'label': 'October', 'value': 10},
                              {'label': 'November', 'value': 11}, {'label': 'December', 'value': 12}],
                     value=7),
        dcc.Dropdown(
            id='start-day-dropdown',
            options=[{'label': i, 'value': i} for i in range(1, 32)],
            placeholder='Select end day:',
            value=9
        ),
        dcc.Dropdown(
            id='start-hour-dropdown',
            options=[{'label': i, 'value': i} for i in range(0, 24)],
            placeholder='Select start hour:',
            value=14
        ),
        dcc.Dropdown(
            id='start-minute-dropdown',
            options=[{'label': i, 'value': i} for i in range(0, 60)],
            placeholder='Select start minute:',
            value=45
        )
    ]),

    # END DATE selector
    html.Div([
        html.H6('Select end date (YEAR, MONTH, DAY, HOUR, MINUTE):'),
        dcc.Input(id = 'endDateYear', value = dt.now().year, type = 'number'),
        dcc.Dropdown(id='endDateMonth',
                     options=[{'label': 'January', 'value': 1}, {'label': 'February', 'value': 2},
                              {'label': 'March', 'value': 3}, {'label': 'April', 'value': 4},
                              {'label': 'May', 'value': 5}, {'label': 'June', 'value': 6},
                              {'label': 'July', 'value': 7}, {'label': 'August', 'value': 8},
                              {'label': 'September', 'value': 9}, {'label': 'October', 'value': 10},
                              {'label': 'November', 'value': 11}, {'label': 'December', 'value': 12}],
                     value=dt.now().month),
        dcc.Dropdown(
            id='end-day-dropdown',
            options=[{'label':i, 'value':i} for i in range(1,32)],
            placeholder='Select end day:',
            value=dt.now().day
        ),
        dcc.Dropdown(
            id='end-hour-dropdown',
            options=[{'label':i, 'value':i} for i in range(0,24)],
            placeholder='Select end hour:',
            value = dt.now().hour
        ),
        dcc.Dropdown(
            id='end-minute-dropdown',
            options=[{'label':i, 'value':i} for i in range(0,60)],
            placeholder='Select end minute:',
            value=dt.now().minute
        )
    ]),

# 3. User selects the months they want to examine over that period (this replaces "summer" and "winter" analyses)
    # dropdown list of months - can select more than one
    html.H5('Step 3: Select months to analyze within the overall range:'),
    dcc.Dropdown(id='monthsToAnalyze',
                 options=[{'label': 'January', 'value': 1}, {'label': 'February', 'value': 2},
                          {'label': 'March', 'value': 3}, {'label': 'April', 'value': 4},
                          {'label': 'May', 'value': 5}, {'label': 'June', 'value': 6},
                          {'label': 'July', 'value': 7}, {'label': 'August', 'value': 8},
                          {'label': 'September', 'value': 9}, {'label': 'October', 'value': 10},
                          {'label': 'November', 'value': 11}, {'label': 'December', 'value': 12}],
                 value = [1,2,3,4,5,6,7,8,9,10,11,12],
                 multi = True
    ),

# 4. User selects the parameter they would like to analyse (temp, RH, or dew point)
    html.H5('Step 4: Select parameter to analyze.'),
    dcc.Dropdown(
        id='parameter-dropdown',
        options=[{'label': 'Temperature', 'value':'Temp'},{'label': 'Relative Humidity','value': 'RH'},
                 {'label': 'Dew Point', 'value': 'DP'}],
        value = 'Temp'
    ),

# 5. User selects the analysis to perform on that parameter (swing or whether it was within bounds)
    html.H5('Step 5: Select the type of analysis to perform on above selected parameter.'),
    html.H6('"Bounds" examines the minimum and maximum of the parameter and "Swing" examines the swing (i.e. the difference'
            ' between the maximum and minimum value of the parameter over a 24 hour period.'),
    dcc.Dropdown(
        id='analysis-dropdown',
        options=[{'label': 'Bounds', 'value': 'BoundsAnalysis'}, {'label': 'Swing', 'value': 'SwingAnalysis'}],
        value = 'BoundsAnalysis'
    ),

# 6. User enters the desired min and max of that parameter (for bounds analysis) or the maximum swing allowed (for
    # swing analysis)
    # default value will appear based on above parameter selected
    html.H5('Step 6: For "Bounds" analysis, enter minimum and maximum bounds on the above selected parameter. '
            'For "Swing" analysis, enter maximum permitted swing.'),
    # minimum input
    dcc.Input(
        id='input_min',
        type='number',
        value = 60,
        placeholder= 'Enter minimum value...'
    ),
    # maximum input
    dcc.Input(
        id='input_max',
        type='number',
        placeholder= 'Enter maximum value...'
    ),

# 7. User can check a box to smooth the data (I'll just have the checkbox there for now and focus on fixing the smoothing code from above after the poster presentation)
    html.H5('Step 7: Check box below to smooth data i.e. have spikes eliminated:'),
    dcc.Checklist(id = 'smoothCheckbox',
        options=[{'label': 'Account for duration', 'value': 'AFD'}],
        value = []
    ),
    # input box shows up if box is checked allowing user to enter max spike length of time to be smoothed
    html.Div(id = 'smooth_explain',
             children = 'Enter the maximum length of spike to be eliminated (For example, entering "3" would remove'
                        ' all spikes less than 30 minutes in length as it would remove spikes less than or equal to '
                        '3 data points apart; data points are 15 minutes apart.).'),
    dcc.Input(
        id='spike_length',
        type='number',
        #placeholder= 'Enter maximum spike length to smooth out...',
        value = 3
    ),


# 8. User hits an "analyse" button
    html.H5('Step 8: Click submit button below to analyze data with the selected above constraints.'),
    html.Button(id='submit-button', n_clicks=0, children='Submit'),

# 9. The interface returns some text detailing the results of the analysis (mins, maxes, number of points that are out
# of bounds, % of time the parameter is out of bounds) and a graph of the data file
    html.H5('Output:'),
    html.Div(id='output-state'),
    dcc.Graph(id='Mygraph'),    # graph of uploaded data file
    html.H6('Contour Plots'),
    html.Div([
        html.Div([
            dcc.Graph(id='contour-max')
        ],className = "six columns"),
        html.Div([
            dcc.Graph(id='contour-min'),
        ],className = "six columns"),
    ],className="row"),

])

# FUNCTIONS_____________________________________________________________________________________________________________
# file content parser; prints name of file parsed
def parse_contents(filename):
    return html.Div([
        # prints filename
        html.H6('You have selected the following file:'),
        html.H6(filename)
    ])

# this function is very specific to Winterthur and their .pm2 files
# creates dataframe of needed form
def parse_data(filename, startDate, endDate, monthsArray):
    # read file into a dataframe, where filename is a 'string'
    df = pd.read_table(filename,skiprows=[0, 1, 3])  # replace specific file with variable from upload
    # set date and time column as datetime type
    df['DATE AND TIME GMT'] = df['DATE AND TIME GMT'].astype('datetime64[ns]')
    # rename columns
    df.columns = ['Date and Time in GMT', 'Temperature (Degrees Fahrenheit)', 'Relative Humidity (%)']

    # select the entered date range, from startDate to endDate
    maskRange = (df['Date and Time in GMT'] > startDate) & (df['Date and Time in GMT'] <= endDate)
    df = df.loc[maskRange]

    # if any specific months are selected, get rid of other months
    if monthsArray != []: # if any month is selected the array will not be all zeros
        df['month'] = df['Date and Time in GMT'].map(lambda x: x.strftime('%m'))  # pulls out month from data column
        df['month'] = pd.to_numeric(df['month'])  # recast month column as ints, not objects
        df = df[df['month'].isin(monthsArray)]

    return df

def add_DP_column(df):
    # if parameter selected is dew point, perform this function on the df
    # calculate dew point; dp = temp - ((100 - RH) / 5)
    #df['Dew Point'] = df['Temperature (Degrees Fahrenheit)'] - ((100 - df['Relative Humidity (%)']) / 5)
    # formula expects temp in degrees C so convert F to C and then convert answer back to F
    #df['Dew Point'] = ((((df['Temperature (Degrees Fahrenheit)'] - 32) * (5 / 9)) - ((100 - df['Relative Humidity (%)'])
                          #  / 5)) * (9 / 5)) + 32
    # convert F to C:
    temp = ((df['Temperature (Degrees Fahrenheit)'] - 32) * (5 / 9))
    RH = df['Relative Humidity (%)']
    # because converted to C, convert back to F at end of eq
    df['Dew Point'] =  ((((RH/100)**(1/8))*(112 + 0.9*temp)+0.1*temp - 112) * (9 / 5)) + 32
    return df

def swing_analysis(df, swingInt,columnName): # columnName is a 'string'
    numEntries = len(df[columnName]) # length of data frame for rh
    pointsInDay = 24 * 60 / 15  # constant; number data points separated in 15 minute increments = 96 points
    lastValue = int(numEntries - pointsInDay - 1)     # calculate last "day", last possible complete 24 hour period

    # initialize blank rhSwing storage array
    swingArray = np.zeros(numEntries)  # could also be a new column in dataframe

    # for the number of possible 24hr periods in the data set, loop; will be zero for end values
    for k in range(0, lastValue):
        jEnd = int(k + pointsInDay)
        # find maximum and minimum value over a one day period
        maxRH = max(df[columnName][k:jEnd])
        minRH = min(df[columnName][k:jEnd])
        # determine swing over 24 hour period and store in array
        swingArray[k] = maxRH - minRH
    df['swing'] = swingArray

    # if swing is greater than permitted, note how many times it is
    sum = 0
    for i in range(0, numEntries):
        if swingArray[i] >= swingInt:
            sum = sum + 1
    # sum = number of times RH Swing is out of bounds
    return swingArray, sum, lastValue, df

## CALLBACKS_____________________________________________________________________________________________________________
# updates graph & analysis based on inputs
@app.callback([Output('Mygraph', 'figure'),
               Output('output-state','children'),
               Output(component_id='contour-max', component_property='figure'),
               Output(component_id='contour-min', component_property='figure')
               ],
            [Input('submit-button','n_clicks')], # only have this because if button is clicked, it will update
              # graph/analysis but value n_clicks is not actually needed anywhere
            [State('upload-data', 'contents'),
            State('upload-data', 'filename'),
            # start date
            State('startDateYear', 'value'),  # input
            State('startDateMonth', 'value'),  # dropdown, '1', '2', etc. for Jan, Feb, etc.
            State('start-day-dropdown', 'value'),  # dropdown, values come as string, corresponding to options
            State('start-hour-dropdown', 'value'),  # dropdown
            State('start-minute-dropdown', 'value'),  # dropdown
            # end date
            State('endDateYear', 'value'),  # input
            State('endDateMonth', 'value'),  # dropdown, '1', '2', etc. for Jan, Feb, etc.
            State('end-day-dropdown', 'value'),  # dropdown, values come as string, corresponding to options
            State('end-hour-dropdown', 'value'),  # dropdown
            State('end-minute-dropdown', 'value'),  # dropdown
            # dropdowns
            State('monthsToAnalyze','value'),
            State('parameter-dropdown','value'),
            State('analysis-dropdown','value'),
            State('input_min','value'),
            State('input_max','value')
            # add in smoothing checkbox later by referencing month selection in dateRangeTest.py for how to do it
            ])
def update_graph__and_analysis(nclicks, contents, filename, startYr, startMonth, startDay, startHr, startMin, endYr,
                               endMonth, endDay, endHr, endMin, monthsArray, parameter, analysis, inputMin, inputMax):
# nclicks is not actually used but it has to be transferred anyways because it is what tells the program to run another
# round of analysis
    if contents:
        filename = filename[0]

        # make date into datetime format
        startDate = dt(startYr, startMonth, startDay, startHr, startMin)
        endDate = dt(endYr, endMonth, endDay, endHr, endMin)

        df = parse_data(filename, startDate, endDate, monthsArray)

        # assign column name according to parameter
        if parameter == 'DP':
            df = add_DP_column(df)
            columnName = 'Dew Point'
        elif parameter == 'Temp':
            columnName = 'Temperature (Degrees Fahrenheit)'
        elif parameter == 'RH':
            columnName = 'Relative Humidity (%)'

        ## the below chunk of code used for making contour plots
        # pull out year and month
        df['year'] = pd.DatetimeIndex(df['Date and Time in GMT']).year
        df['month'] = pd.DatetimeIndex(df['Date and Time in GMT']).month
        # make arrays of years and months (just unique values)
        yearsArray = np.unique(df['year'])
        #monthsArray = np.unique(df['month'])
        # create storage arrays
        storageMin = np.zeros([len(monthsArray), len(yearsArray)])
        storageMax = np.zeros([len(monthsArray), len(yearsArray)])

        if analysis == 'BoundsAnalysis':
            # determine max and min
            maxValueB = round(max(df[columnName]),2)
            minValueB = round(min(df[columnName]),2)
            # determine how often the data goes out of bounds
            # too low:
            numLow = np.sum(df[columnName] <= inputMin)

            # LATER ON why numCalcs and not numEntries in contour reports for calcing percentage of entries that are off?
            # i chose to use numEntries for now
            numEntries = len(df[columnName])
            percentLow = round((numLow / numEntries) * 100,2)
            # too high:
            numHigh = np.sum(df[columnName] >= inputMax)
            numEntries = len(df[columnName])
            percentHigh = round((numHigh / numEntries) * 100,2)
            # out of bounds in general:
            percentOutOfBounds = round(percentLow + percentHigh,2)
            # value to return to text output:
            children = u''' The maximum value of {} is {} and the minimum value is {}.
            {} was less than the desired minimum {}% of the time.
            {} was greater than the desired maximum {}% of the time.
            Overall, {} was out of the desired bounds {}% of the time.
            '''.format(columnName, maxValueB,minValueB, columnName, percentLow, columnName, percentHigh, columnName,
                       percentOutOfBounds)

            # Add bound lines to graph
            df['Lower Bound'] = inputMin
            df['Upper Bound'] = inputMax

            # graph to return
            figure = {'data': [{'x': df['Date and Time in GMT'], 'y': df[columnName], 'name': columnName},
                               {'x': df['Date and Time in GMT'], 'y': df['Lower Bound'],'name': 'Lower Bound'},
                               {'x': df['Date and Time in GMT'], 'y': df['Upper Bound'], 'name': 'Upper Bound'}
                              ],
                      'layout': {'title': columnName}}

            # loop, creating storage array, which is the array of values to be plotted for each (month,year) data point in plot
            for i in range(0, len(monthsArray)):
                for j in range(0, len(yearsArray)):
                    # creates boolean array of
                    totalPoints = np.sum((df['year'] == yearsArray[j]) & (df['month'] == monthsArray[i]))
                    outOfRangePoints = np.sum((df['year'] == yearsArray[j]) & (df['month'] == monthsArray[i]) &
                                              (df[columnName] <= inputMin))
                    storageMin[i, j] = (outOfRangePoints / totalPoints) * 100

            # create figure to send to interface
            figMin = go.Figure(data=
            go.Contour(
                z=storageMin,
                x=yearsArray,  # horizontal axis
                y=monthsArray, # vertical axis
                colorscale='GnBu'
                #colorscale = colorscale1
            ))
            figMin.update_layout(
                title="% of points where {} is below the desired minimum".format(columnName),
                xaxis_title="Years",
                yaxis_title="Months"
            )

            # loop, creating rhSwingMat which is the array of values to be plotted for each (month,year) data point in plot
            for i in range(0, len(monthsArray)):
                for j in range(0, len(yearsArray)):
                    # creates boolean array of
                    totalPoints = np.sum((df['year'] == yearsArray[j]) & (df['month'] == monthsArray[i]))
                    outOfRangePoints = np.sum((df['year'] == yearsArray[j]) & (df['month'] == monthsArray[i]) &
                                              (df[columnName] >= inputMax))
                    storageMax[i, j] = (outOfRangePoints / totalPoints) * 100

            # create figure to send to interface
            figMax = go.Figure(data=
            go.Contour(
                z=storageMax,
                x=yearsArray,  # horizontal axis
                y=monthsArray  # vertical axis
                ,
                colorscale='GnBu',
                #colorscale='Hot',
                #colorscale = colorscale2,
                contours=dict(
                    start=0,
                    end=np.max(storageMax),
                    size=2)
            ))

            figMax.update_layout(
                title="% of points where {} is above the desired maximum".format(columnName),
                xaxis_title="Years",
                yaxis_title="Months"
            )

            return figure, children, figMin, figMax


        elif analysis == 'SwingAnalysis':
            # perform analysis
            swingArray, sum, lastValue, df = swing_analysis(df, inputMax, columnName)
            # max swing
            maxValueS = round(max(swingArray),2)
            # determine percent of time out of bounds
            numSwing = np.sum(swingArray >= inputMax)
            numEntries = len(swingArray)
            percentSwing = round(numSwing / numEntries * 100, 2)
            # text to output:
            children = u''' The maximum swing value for {} is {} and there were {} times that {} was out of the 
            permitted swing range of {}. The {} swing was out bounds {}% of the time.
            '''.format(columnName, maxValueS, sum, columnName, inputMax, columnName, percentSwing)
            # graph to return
            figure = {'data': [{'x': df['Date and Time in GMT'], 'y': df[columnName], 'name': columnName}],
                'layout':{'title': columnName}}

            # loop, creating rhSwingMat which is the array of values to be plotted for each (month,year) data point in plot
            for i in range(0, len(monthsArray)):
                for j in range(0, len(yearsArray)):
                    # creates boolean array of
                    totalPoints = np.sum((df['year'] == yearsArray[j]) & (df['month'] == monthsArray[i]))
                    outOfRangePoints = np.sum((df['year'] == yearsArray[j]) & (df['month'] == monthsArray[i]) &
                                              (df['swing'] >= inputMax))
                    storageMax[i, j] = (outOfRangePoints / totalPoints) * 100

            # create figure to send to interface
            figMax = go.Figure(data=
            go.Contour(
                z=storageMax,
                x=yearsArray,  # horizontal axis
                y=monthsArray  # vertical axis
                ,
                #colorscale='Hot',
                colorscale='GnBu',
                contours=dict(
                    start=0,
                    end=np.max(storageMax),
                    size=2)
            ))

            figMax.update_layout(
                title="% of points where {} swing is greater than the desired swing value".format(columnName),
                xaxis_title="Years",
                yaxis_title="Months"
            )
            # only need one graph for swing
            figMin = {'data': [{'x': [0], 'y': [0], 'name': 'N/A'}, ],
                      'layout': {'title': 'See graph at right for swing.'}}

            return figure, children, figMin, figMax

    else:
        figure =  {'data': [{'x': [0], 'y': [0], 'name':'N/A'},],'layout':
            {'title': 'No data uploaded yet'} }
        children = 'Click "Submit" button to run analysis and determine maximum and minimum values.'
        return figure, children, figure, figure

# data file selector callback; displays name of file uploaded
@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'filename')])
def update_output(list_of_names):
    if list_of_names is not None:
        children = [
            parse_contents(n) for n in
            zip(list_of_names)]
        return children

# show/hide bound input boxes based on type of analysis being performed
@app.callback(Output(component_id='input_min', component_property='style'),
             [Input(component_id='analysis-dropdown', component_property='value')])
def show_hide_element(analysis):
    if analysis == 'SwingAnalysis':
        return {'display': 'none'}

# change suggested bound values depending on parameter and type of analysis
@app.callback([Output(component_id='input_min', component_property='value'),
               Output(component_id='input_max', component_property='value')],
              [Input(component_id='parameter-dropdown', component_property= 'value'),
               Input(component_id='analysis-dropdown', component_property='value')]
              )
def change_value2(parameter, analysis):
    if analysis == 'SwingAnalysis':
        return 0, 10
    else:
        if parameter == 'Temp':
            return 63, 74
        elif parameter == 'RH':
            return 35, 57
        elif parameter == 'DP':
            return 37, 56

@app.callback(Output(component_id='spike_length', component_property='style'),
              [Input(component_id='smoothCheckbox', component_property='value')])
def show_hide_element(visibility_state):
    if visibility_state == []:
        return {'display': 'none'}

@app.callback(Output(component_id='smooth_explain', component_property='style'),
              [Input(component_id='smoothCheckbox', component_property='value')])
def show_hide_element(visibility_state):
    if visibility_state == []:
        return {'display': 'none'}

if __name__ == '__main__':
   # app.run_server(debug = 'False',host = '0.0.0.0', port = 8080)#debug=True) #host = '0.0.0.0')
    app.run_server(debug = 'True')
#!/usr/bin/env python
# coding: utf-8

# ## Problem: To schedule and return the most optimised solution for scheduling the 6 jobs carried out by 4 machines where there is only 1 machine working on a single job at a time and the teams can start working on a different job immediately

# |   Job    | Machine 1 | Machine 2 | Machine 3 | Machine 4 |
# |----------|--------------|----------|---------|----------|
# |  Task 1  |      60      |    -     |    75   |    20    |
# | Task 2 |      70      |    -     |    20   |   50    |
# |   Task 3  |      60      |    20    |    80   |    40    |
# |   Task 4    |      90      |   100    |   120   |   120    |
# |  Task 5  |      80      |    30    |    45   |    60    |
# |   Task 6   |      -       |   120    |    30   |    -     |
# 

# ### Constraints: 
# •	The machine 1 need to finish before the machine 2 can start, \
# •	The machine 2 need to finish before the machine 3 can start, and \
# •	The machine 3 need to finish before the machine 4 can start.
# 
# 

# ## Problem Formulation: 
# ### Constraint Programming (CP):
# Constraint Programming (CP) is utilized as the primary approach to solve the scheduling problem.
# CP is a declarative programming paradigm where **relationships between variables are expressed as constraints**. 
# 
# The **OR-Tools library**, specifically the CP-SAT solver, is employed to model and solve the optimization problem. 
# 
# ## Flowchart of Constraint Programming:
# ### Import Libraries &rarr; Declare model &rarr; Create Variables &rarr; Create Constraints &rarr; Call Solver &rarr; Plot
# 
# 
# The input data, representing jobs, teams, and processing times, is organized into a structured format. **Pandas**, a powerful data manipulation library in Python, was used to handle the dataset efficiently. 
# 
# Tasks within **each job are represented as intervals with start and end times**, constrained by the processing times and team availability.
# 
# The CP model is passed to the CP-SAT solver for optimization.
# The **solver iteratively explores the search space to find an optimal solution that satisfies all constraints.** 
# 
# Once an optimal solution is found, the assigned tasks are visualized using a **Gantt chart**.
# **Plotly's figure factory (ff)** library is utilized to create the Gantt chart, providing a clear visualization of the schedule.

# ## Solution:

# ### Import Libraries

# In[1]:


import pandas as pd
import collections
from ortools.sat.python import cp_model
import plotly.figure_factory as ff


# #### This code defines a function `visualize_schedule` that generates a schedule visualization by mapping assigned jobs to specific machines with corresponding start and finish times, based on a given plan date.

# In[2]:


plan_date = pd.to_datetime('02/28/2023 08:00:00')

def visualize_schedule(assigned_jobs, all_teams, plan_date):
    final = []
    task_labels = {0: 'Task 1', 1: 'Task 2', 2: 'Task 3', 3: 'Task 4', 4: 'Task 5', 5: 'Task 6'}
    machine_labels = {0: 'Machine 1', 1: 'Machine 2', 2: 'Machine 3', 3: 'Machine 4'}
    
    for team in all_teams:
        assigned_jobs[team].sort()
        for assigned_task in assigned_jobs[team]:
            task_name = task_labels[assigned_task.job]
            team_name = machine_labels[team]
            temp = dict(Task=task_name, Start=plan_date + pd.DateOffset(minutes=assigned_task.start),
                        Finish=plan_date + pd.DateOffset(minutes=(assigned_task.start + assigned_task.duration)),
                        Resource=team_name)
            final.append(temp)
    final.sort(key=lambda x: x['Task'])
    return final


# ### Data pre-processing
# #### Create the dataframe

# In[3]:


data = {
    'Job': ['Task 1', 'Task 1', 'Task 1', 'Task 2', 'Task 2', 'Task 2', 'Task 3', 'Task 3', 'Task 3', 'Task 3', 'Task 4', 'Task 4', 'Task 4', 'Task 4', 'Task 5', 'Task 5', 'Task 5', 'Task 5', 'Task 6', 'Task 6'],
    'Machine': ['Machine 1', 'Machine 3', 'Machine 4', 'Machine 1', 'Machine 3', 'Machine 4', 'Machine 1', 'Machine 2', 'Machine 3', 'Machine 4', 'Machine 1', 'Machine 2', 'Machine 3', 'Machine 4', 'Machine 1', 'Machine 2', 'Machine 3', 'Machine 4', 'Machine 2', 'Machine 3'],
    'Team_ID': [0, 2, 3, 0, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2],
    'Processing Time (in minutes)': [60, 75, 20, 70, 20, 50, 60, 20, 80, 40, 90, 100, 120, 120, 80, 30, 45, 60, 120, 30]
}

# Convert the dataset into a DataFrame
jobs_df = pd.DataFrame(data)
print(jobs_df)


# #### This code segment creates a list of tuples for each job by selecting the team ID and processing time from the `jobs_df` DataFrame based on the job. Then, it organizes these tuples into nested lists representing all the jobs. It calculates the number of machines needed for the jobs and calculates the horizon, which represents the total time span required to complete all tasks across all jobs.

# In[4]:


# Create a list of tuples for each job
job1 = list(jobs_df.loc[jobs_df['Job'] == 'Task 1'][['Team_ID', 'Processing Time (in minutes)']].itertuples(index=False, name=None))
job2 = list(jobs_df.loc[jobs_df['Job'] == 'Task 2'][['Team_ID', 'Processing Time (in minutes)']].itertuples(index=False, name=None))
job3 = list(jobs_df.loc[jobs_df['Job'] == 'Task 3'][['Team_ID', 'Processing Time (in minutes)']].itertuples(index=False, name=None))
job4 = list(jobs_df.loc[jobs_df['Job'] == 'Task 4'][['Team_ID', 'Processing Time (in minutes)']].itertuples(index=False, name=None))
job5 = list(jobs_df.loc[jobs_df['Job'] == 'Task 5'][['Team_ID', 'Processing Time (in minutes)']].itertuples(index=False, name=None))
job6 = list(jobs_df.loc[jobs_df['Job'] == 'Task 6'][['Team_ID', 'Processing Time (in minutes)']].itertuples(index=False, name=None))

# Create nested list for all the sales orders
jobs_data = [job1, job2, job3, job4, job5, job6]

# Get the number of teams
teams_count = len(jobs_df['Machine'].unique())
all_teams = range(teams_count)

# Calculate the horizon as the sum of all durations
horizon = sum(jobs_df['Processing Time (in minutes)'])


# ### Declare model
# #### This code segment creates a CP model using the OR-Tools library. It defines named tuples to store information about variables representing task start and end times. It iterates over each job and task within the `jobs_data`, creating interval variables for each task and adding them to corresponding team lists based on the assigned machine. It constructs a dictionary `all_tasks` to store information about created variables.

# In[5]:


# Create the model.
model = cp_model.CpModel()

# Named tuple to store information about created variables.
task_type = collections.namedtuple('task_type', 'start end interval')
# Named tuple to manipulate solution information.
assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')


# ### Create Variables

# In[6]:


# Creates job intervals and add to the corresponding team lists.
all_tasks = {}
machine_to_intervals = collections.defaultdict(list)

for job_id, job in enumerate(jobs_data):
    for task_id, task in enumerate(job):
        machine = task[0]
        duration = task[1]
        suffix = '_%i_%i' % (job_id, task_id)
        start_var = model.NewIntVar(0, horizon, 'start' + suffix)
        end_var = model.NewIntVar(0, horizon, 'end' + suffix)
        interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
        all_tasks[job_id, task_id] = task_type(start=start_var,
                                                   end=end_var,
                                                   interval=interval_var)
        machine_to_intervals[machine].append(interval_var)


# ### Create constraints
# #### This code segment adds constraints to the CP model to enforce that when a team finishes a task within a job, they may start the next task immediately. It iterates over each job and task within the `jobs_data`, ensuring that the end time of the current task is before the start time of the next task within the same job. Additionally, it creates and adds disjunctive constraints to ensure that tasks assigned to the same machine do not overlap in time. Finally, it defines the makespan objective to minimize the completion time of the last task in all jobs and adds it to the model for optimization.

# In[7]:


job_id


# In[8]:


# Add constraint: when a team finishes a job, they may start another job immediately
for job_id, job in enumerate(jobs_data):
    for task_id in range(len(job) - 1):  # Iterate up to the second last task
        machine = job[task_id][0]
        end_var = all_tasks[job_id, task_id].end
        next_start_var = all_tasks[job_id, task_id + 1].start
        model.Add(next_start_var >= end_var)



# Create and add disjunctive constraints.
for team in all_teams:
    model.AddNoOverlap(machine_to_intervals[team])

# Makespan objective.
obj_var = model.NewIntVar(0, horizon, 'makespan')
model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)])
model.Minimize(obj_var)


# ### Call Solver
# #### This code segment creates a solver object using the CP-SAT solver from OR-Tools and solves the CP model. It checks if the solver found an optimal solution and prints the optimal schedule length if one was found. If no solution was found, it prints a message indicating that no solution was found. Additionally, it constructs a dictionary `assigned_jobs` to store the assigned tasks per machine, including their start times, job IDs, task indices, and durations.

# In[9]:


# Creates the solver and solve.
solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL:
    print('Solution:')
    # Create one list of assigned tasks per machine.
    assigned_jobs = collections.defaultdict(list)
    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            assigned_jobs[machine].append(
                assigned_task_type(start=solver.Value(
                    all_tasks[job_id, task_id].start),
                                    job=job_id,
                                    index=task_id,
                                    duration=task[1]))

    # Finally print the solution found.
    print(f'Optimal Schedule Length: {solver.ObjectiveValue()}')


# ### Plot
# #### Visualise the schedule using a Gantt chart

# In[10]:


res = visualize_schedule(assigned_jobs, all_teams, plan_date)
fig = ff.create_gantt(res, index_col='Resource', show_colorbar=True, group_tasks=True)
fig.show()


# In[ ]:





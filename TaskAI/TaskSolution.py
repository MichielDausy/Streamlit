# %% [markdown]
# ## Cryptarithmetic Puzzle
# 
# First I import the neccesary libraries to complete this task

# %%
import streamlit as st
from simpleai.search import CspProblem, backtrack

# %% [markdown]
# The streamlit app asks the user to enter 3 words
# 
# The words are concatenated, placed in a set so there are no duplicate characters and then placed into a tuple

# %%
number1 = st.text_input("Enter the first word:") #TO
number2 = st.text_input("Enter the second word:") #GO
result = st.text_input("Enter the result:") #OUT
variables = tuple(set(number1+number2+result)) #TOGU

# %%
with st.form(key='my_form_to_submit'):
    submit_button = st.form_submit_button(label='Submit')

# %% [markdown]
# I then set the possible values for all the characters that the user entered where the first character of number 1, 2 and result cannot be 0
# 
# The other chracters can be a number from 0 to 10

# %%
#if len(number1) > 0 and len(number2) > 0 and len(result) > 0:
if submit_button:
        domains = {
                number1[0]: list(range(1, 10)),
                number2[0]: list(range(1, 10)),
                result[0]: list(range(1, 10)),
        }

# %% [markdown]
# The other characters aren't added to the domain dictionary yet
# 
# To do this I use a for loop to dynamically add the possible values to a character which are the numbers 0 through 10

# %%
if submit_button:
    for variable in variables:
        if variable not in domains:
            domains[variable] = list(range(0, 10))

# %% [markdown]
# Here I add a constraint to add the 2 words together

# %%
def constraint_unique(variables, values):
    return len(values) == len(set(values))  # remove repeated values and count

def constraint_add(variables, values):
    factor1 = ""
    factor2 = ""
    sum = ""
    for char in number1:
        factor1 += str(values[variables.index(char)])
    for char in number2:
        factor2 += str(values[variables.index(char)])
    for char in result:
        sum += str(values[variables.index(char)])
    return (int(factor1) + int(factor2)) == int(sum)

# %%
constraints = [
    (variables, constraint_unique), #TOGU
    (variables, constraint_add), #TOGU
]

# %%
#if len(number1) > 0 and len(number2) > 0 and len(result) > 0:
if submit_button:
    problem = CspProblem(variables, domains, constraints)
    output = backtrack(problem)
    print('\nSolutions:', output)
else:
    output = None
    print('No solution')

# %% [markdown]
# Compute the number of columns necessary

# %%
if submit_button:
    n_col = len(result) + 1
    cols = st.columns(n_col)
    rows = 3

# %%
letters = []
numbers = []
if output is not None:
    for letter, number in output.items():
        letters.append(letter)
        numbers.append(str(number)) #ik maak een string van de nummers zodat de spacing in de output overeenkomt met de letters
    st.text(letters)
    st.text(numbers)
    #---------------------------------
    st.write(number1)
    st.write("&plus;", number2)
    st.write(result)
    #---------------------------------
    letters1 = ""
    letters2 = ""
    letters3 = ""
    for letter in number1:
        letters1 += str(output[letter])
    for letter in number2:
        letters2 += str(output[letter])
    for letter in result:
        letters3 += str(output[letter])
    st.write(letters1)
    st.write("&plus;", letters2)
    st.write(letters3)

# %% [markdown]
# ## Generative AI Tools
# 
# ### Prompts used
# 
# In python can I loop over a list to creatre a key value pair in a dictionary - BingAI
# 
# In python using the simpleai library explain how the constraints work - BingAI
# 
# in streamlit when assigning a user input to a variable can i assign a default value if the user doesn't enter a value - BingAI
# 
# how do i make a grid in streamlit that grows dynamically with the length of a number where every individual number has its own column - BingAI
# 
# 



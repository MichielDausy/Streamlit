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
variables = (set(number1+number2+result)) #TOGU

# %% [markdown]
# I then set the possible values for all the characters that the user entered where the first character of number 1, 2 and result cannot be 0
# 
# The other chracters can be a number from 0 to 10

# %%
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
for variable in variables:
    domains[variable] = list(range(0,10))

# %% [markdown]
# Here I add a constraint to add the 2 words together 

# %%
def constraint_unique(variables, values):
    return len(values) == len(set(values))  # remove repeated values and count

def constraint_add(variables, values):
    factor1 = ""
    factor2 = ""
    result = ""
    for char in number1:
        factor1 += values[variables.index(char)]
    for char in number2:
        factor2 += values[variables.index(char)]
    for char in result:
        result += values[variables.index(char)]
    return (factor1 + factor2) == result

# %%
constraints = [
    (variables, constraint_unique),
    (variables, constraint_add),
]

# %%
problem = CspProblem(variables, domains, constraints)

output = backtrack(problem)
print('\nSolutions:', output)

# %% [markdown]
# ## Generative AI Tools
# 
# ### Prompts used
# 



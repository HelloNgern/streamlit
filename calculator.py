import streamlit as st

st.title('Simple Calculator')

# Number inputs for two values
num1 = st.number_input('Enter first number:', value=0.0)
num2 = st.number_input('Enter second number:', value=0.0)

# Operation selector
operation = st.selectbox('Select operation:', ['Add', 'Subtract', 'Multiply', 'Divide'])

# Calculate button
if st.button('Calculate'):
    if operation == 'Add':
        result = num1 + num2
    elif operation == 'Subtract':
        result = num1 - num2
    elif operation == 'Multiply':
        result = num1 * num2
    elif operation == 'Divide':
        if num2 != 0:
            result = num1 / num2
        else:
            st.error('Cannot divide by zero!')
            result = None
    
    if result is not None:
        st.success(f'Result: {result}')
import streamlit as st


class MultiPage:
    """ 
    Framework for combining multiple streamlit applications
    """

    def __init__(self) -> None:
        '''
        Constructor class to generate a list of pages
        '''
        self.pages = []

    def add_page(self, title, func) -> None:
        '''
        Add a page to the list of pages

        Parameters
        ----------
        title : str
            Title of the page
        funct : function
            Function to run when the page is selected
        '''
        self.pages.append({
            'title': title,
            'function': func
        })
    
    # def run(self):
    #     '''
    #     Dropdown menu to select a page
    #     '''
    #     st.sidebar.title('Pages')
    #     page_name = st.sidebar.selectbox(
    #         'Select a page',
    #         [p['title'] for p in self.pages])
    #     page = [p for p in self.pages if p['title'] == page_name][0]
    #     page['function']()
    def run(self):
        # Drodown to select the page to run  
        page = st.sidebar.selectbox(
            'App Navigation', 
            self.pages, 
            format_func=lambda page: page['title']
        )

        # run the app function 
        page['function']()
import pandas as pd
from habanero import Crossref


def find_doi(title):
    """Searches Crossref for a DOI for a given title.

    :param title: Title of the article
    :return: DOI if found else None
    """
    try:
        cr = Crossref()
        return cr.works(query=title, limit=1)['message']['items'][0]['DOI']
    except:
        print(title)
        return None


def find_license(doi):
    """Searches Crossref for the license given a DOI.
    
    :param doi: DOI of the article
    :return: license if available else None 
    """
    try:
        cr = Crossref()
        return cr.works(doi)['message']['license'][0]['URL']
    except:
        print(doi)
        return None
    
    
if __name__ == '__main__':
    # Read file of search results containing title, abstracts and other meta information
    df = pd.read_excel('../data/combined scopus_web of science_mit doi.xlsx').reset_index(drop=True)
    df = df.rename(columns={'Abstract':'text', 'Article Title':'title'})
    
    # Search for DOI if it is not given
    df.loc[df['DOI'].isna(), 'DOI'] = df[df['DOI'].isna()]['title'].apply(find_doi)
    df = df[df['DOI'].notna()]
    df['doi_link'] = f'<a href="{'https://doi.org/' + df['DOI']}">{'https://doi.org/' + df['DOI']}</a>'
    df.to_pickle('../data/df_abstracts.pckl')
    
    # Add license to articles
    df['license'] = df['DOI'].apply(find_license)
    allowed_licenses = pd.read_excel('../data/allowed_licenses.xlsx', header=None)[0].to_list()
    df = df[df['license'].isin(allowed_licenses)]
    df.to_pickle('../data/df_allowed_license.pckl')
    



    
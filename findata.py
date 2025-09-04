import requests
from bs4 import BeautifulSoup
import pandas as pd
from fuzzywuzzy import fuzz, process
import re
from urllib.parse import urljoin, urlparse
import time
from typing import List, Dict, Tuple, Set
import json

class CompanyMatcher:
    def __init__(self, company_list: List[str]):
        """
        Initialize with your list of companies
        """
        self.company_list = [company.strip() for company in company_list]
        self.scraped_companies = set()
        self.matches = []
        
    def clean_company_name(self, name: str) -> str:
        """Clean and normalize company names for better matching"""
        # Remove common corporate suffixes and clean the name
        suffixes = [
            r'\s+Ltd\.?$', r'\s+Limited$', r'\s+Inc\.?$', r'\s+Corp\.?$', 
            r'\s+Corporation$', r'\s+Co\.?$', r'\s+Company$', r'\s+SA$',
            r'\s+Holdings$', r'\s+Group$', r'\s+PLC$', r'\s+LLC$'
        ]
        
        cleaned = name.strip()
        for suffix in suffixes:
            cleaned = re.sub(suffix, '', cleaned, flags=re.IGNORECASE)
        
        # Remove extra whitespace and convert to lowercase for comparison
        return ' '.join(cleaned.split()).lower()
    
    def scrape_webpage(self, url: str, delay: float = 1.0) -> Set[str]:
        """
        Scrape a webpage and extract potential company names
        """
        print(f"Scraping: {url}")
        companies_found = set()
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get all text content
            text_content = soup.get_text()
            
            # Look for company names in various formats
            # This regex looks for potential company names (capitalized words)
            potential_companies = re.findall(r'\b[A-Z][a-zA-Z\s&\-\.]+(?:Ltd|Limited|Inc|Corp|Corporation|Co|Company|SA|Holdings|Group|PLC|LLC)\.?\b', text_content)
            
            # Also look for companies in lists, tables, etc.
            for element in soup.find_all(['li', 'td', 'th', 'div', 'span', 'p']):
                text = element.get_text().strip()
                if text and len(text) > 3 and len(text) < 200:  # Reasonable length for company names
                    # Check if it looks like a company name
                    if any(suffix in text.lower() for suffix in ['ltd', 'inc', 'corp', 'company', 'co.', 'group', 'holdings']):
                        potential_companies.append(text)
            
            # Clean and add to our set
            for company in potential_companies:
                cleaned = company.strip()
                if len(cleaned) > 5:  # Filter out very short matches
                    companies_found.add(cleaned)
            
            print(f"Found {len(companies_found)} potential companies on this page")
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
        
        time.sleep(delay)  # Be respectful to the server
        return companies_found
    
    def find_matches(self, scraped_companies: Set[str], similarity_threshold: int = 80) -> List[Dict]:
        """
        Find matches between your company list and scraped companies
        """
        matches = []
        
        # Clean both lists for comparison
        cleaned_your_companies = {self.clean_company_name(company): company for company in self.company_list}
        cleaned_scraped = {self.clean_company_name(company): company for company in scraped_companies}
        
        # Find exact matches first
        for cleaned_name, original_name in cleaned_your_companies.items():
            if cleaned_name in cleaned_scraped:
                matches.append({
                    'your_company': original_name,
                    'scraped_company': cleaned_scraped[cleaned_name],
                    'match_type': 'exact',
                    'similarity_score': 100
                })
        
        # Find fuzzy matches for remaining companies
        matched_your_companies = {match['your_company'] for match in matches}
        remaining_your_companies = [comp for comp in self.company_list if comp not in matched_your_companies]
        
        for your_company in remaining_your_companies:
            cleaned_your = self.clean_company_name(your_company)
            
            # Find best match using fuzzy matching
            best_match = process.extractOne(
                cleaned_your, 
                list(cleaned_scraped.keys()), 
                scorer=fuzz.ratio
            )
            
            if best_match and best_match[1] >= similarity_threshold:
                matches.append({
                    'your_company': your_company,
                    'scraped_company': cleaned_scraped[best_match[0]],
                    'match_type': 'fuzzy',
                    'similarity_score': best_match[1]
                })
        
        return matches
    
    def scrape_multiple_urls(self, urls: List[str], delay: float = 1.0) -> None:
        """
        Scrape multiple URLs and collect all company names
        """
        all_companies = set()
        
        for url in urls:
            companies = self.scrape_webpage(url, delay)
            all_companies.update(companies)
        
        self.scraped_companies = all_companies
        print(f"\nTotal unique companies found across all pages: {len(all_companies)}")
    
    def analyze_and_report(self, similarity_threshold: int = 80) -> pd.DataFrame:
        """
        Analyze matches and create a detailed report
        """
        if not self.scraped_companies:
            print("No companies scraped yet. Please run scrape_multiple_urls first.")
            return pd.DataFrame()
        
        matches = self.find_matches(self.scraped_companies, similarity_threshold)
        
        if matches:
            df = pd.DataFrame(matches)
            df = df.sort_values('similarity_score', ascending=False)
            
            print(f"\n=== MATCH ANALYSIS ===")
            print(f"Total companies in your list: {len(self.company_list)}")
            print(f"Total companies scraped: {len(self.scraped_companies)}")
            print(f"Total matches found: {len(matches)}")
            print(f"Match rate: {len(matches)/len(self.company_list)*100:.1f}%")
            
            print(f"\nBy match type:")
            match_type_counts = df['match_type'].value_counts()
            for match_type, count in match_type_counts.items():
                print(f"  {match_type.title()} matches: {count}")
            
            print(f"\n=== TOP MATCHES ===")
            print(df.head(10).to_string(index=False))
            
            return df
        else:
            print("No matches found.")
            return pd.DataFrame()
    
    def save_results(self, df: pd.DataFrame, filename: str = "cdp_company_matches.csv"):
        """Save results to CSV"""
        if not df.empty:
            df.to_csv(filename, index=False)
            print(f"\nResults saved to {filename}")
    
    def get_unmatched_companies(self) -> List[str]:
        """Get companies from your list that weren't matched"""
        if not self.matches:
            return self.company_list
        
        matched_companies = {match['your_company'] for match in self.matches}
        return [company for company in self.company_list if company not in matched_companies]

# Example usage
if __name__ == "__main__":
    # Your company list (from the uploaded document)
    your_companies = [
        "China Oilfield Services Ltd.",
        "China Petroleum Engineering Corp.",
        "CNOOC Energy Technology & Services Ltd.",
        "Dalipal Holdings Ltd.",
        "MODEC",
        "Offshore Oil Engineering Co.",
        "RAIZNEXT Corp.",
        "Shandong Xinchao Energy Corp. Ltd.",
        # ... (add all your companies here)
    ]
    
    # Initialize the matcher
    matcher = CompanyMatcher(your_companies)
    
    # URLs to scrape - you can add more CDP pages here
    urls_to_scrape = [
        "https://www.cdp.net/en/data/scores",
        "https://www.cdp.net/en/press-releases/cdp-a-list-2024",
        # Add more CDP URLs as needed
    ]
    
    # Scrape the websites
    print("Starting web scraping...")
    matcher.scrape_multiple_urls(urls_to_scrape, delay=1.0)
    
    # Analyze and report matches
    results_df = matcher.analyze_and_report(similarity_threshold=75)
    
    # Save results
    if not results_df.empty:
        matcher.save_results(results_df)
    
    # Show unmatched companies
    unmatched = matcher.get_unmatched_companies()
    if unmatched:
        print(f"\n=== UNMATCHED COMPANIES ===")
        for company in unmatched[:20]:  # Show first 20
            print(f"  {company}")
        if len(unmatched) > 20:
            print(f"  ... and {len(unmatched) - 20} more")

    print("\n=== USAGE INSTRUCTIONS ===")
    print("1. Replace 'your_companies' list with your full company list")
    print("2. Add more CDP URLs to 'urls_to_scrape' list")  
    print("3. Adjust similarity_threshold (75-90 recommended)")
    print("4. Run the script to get matches and similarity analysis")
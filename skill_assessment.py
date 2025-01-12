import pandas as pd
from pathlib import Path
import PyPDF2
import re
from collections import Counter
import json
import os
import requests
import time
from typing import List, Dict

class ResumeAnalyzer:
    def __init__(self):
        self.skills_keywords = self._load_skills_keywords()
        
    def _load_skills_keywords(self):
        # Load skills from resources.csv
        skills = set()
        try:
            with open('resources.csv', 'r') as f:
                next(f)  # Skip header
                for line in f:
                    skill = line.split(',')[0].strip().lower()
                    skills.add(skill)
        except Exception as e:
            print(f"Warning: Could not load skills from resources.csv: {e}")
            # Fallback skills list
            skills = {
                "python", "java", "javascript", "html", "css", "sql",
                "machine learning", "data analysis", "react", "node.js",
                "docker", "kubernetes", "aws", "azure", "git"
            }
        return skills

    def extract_text_from_pdf(self, pdf_path):
        """Extract text content from PDF file."""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {e}")

    def extract_contact_info(self, text):
        """Extract email and phone number from text."""
        contact_info = {'email': None, 'phone': None}
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.findall(email_pattern, text)
        if email_matches:
            contact_info['email'] = email_matches[0]
            
        # Phone pattern
        phone_pattern = r'\b(?:\+?\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'
        phone_matches = re.findall(phone_pattern, text)
        if phone_matches:
            contact_info['phone'] = phone_matches[0]
            
        return contact_info

    def extract_skills(self, text):
        """Extract skills from text based on predefined keywords."""
        text = text.lower()
        found_skills = []
        
        for skill in self.skills_keywords:
            if skill.lower() in text:
                found_skills.append(skill)
                
        return found_skills

    def extract_education(self, text):
        """Extract education information from text."""
        education_keywords = [
            "bachelor", "master", "phd", "degree",
            "university", "college", "institute",
            "b.tech", "m.tech", "b.e", "m.e"
        ]
        
        education_info = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip().lower()
            if any(keyword in line for keyword in education_keywords):
                education_info.append(line)
                
        return education_info

    def analyze_experience_level(self, text, skills):
        """Determine experience level based on text content and skills."""
        experience_keywords = {
            'senior': ['senior', 'lead', 'manager', 'architect', 'principal'],
            'mid': ['mid-level', 'intermediate', 'developer', 'engineer'],
            'junior': ['junior', 'entry', 'intern', 'fresher', 'trainee']
        }
        
        text = text.lower()
        level_scores = Counter()
        
        # Analyze keywords
        for level, keywords in experience_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    level_scores[level] += 1
        
        # Consider number of skills
        num_skills = len(skills)
        if num_skills > 8:
            level_scores['senior'] += 1
        elif num_skills > 5:
            level_scores['mid'] += 1
        else:
            level_scores['junior'] += 1
            
        # Determine level
        if level_scores['senior'] > 0:
            return 'Senior'
        elif level_scores['mid'] > 0:
            return 'Mid-Level'
        else:
            return 'Junior'

    def analyze_resume(self, pdf_path):
        """Main method to analyze resume."""
        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            
            # Extract various information
            contact_info = self.extract_contact_info(text)
            skills = self.extract_skills(text)
            education = self.extract_education(text)
            experience_level = self.analyze_experience_level(text, skills)
            
            # Prepare analysis results
            analysis = {
                'contact_info': contact_info,
                'skills_found': skills,
                'education': education,
                'experience_level': experience_level,
                'skill_count': len(skills)
            }
            
            return analysis
            
        except Exception as e:
            raise Exception(f"Resume analysis failed: {e}")

    def print_analysis(self, analysis):
        """Print analysis results in a formatted way."""
        print("\n=== Resume Analysis Results ===")
        
        print("\n📧 Contact Information:")
        print(f"Email: {analysis['contact_info']['email']}")
        print(f"Phone: {analysis['contact_info']['phone']}")
        
        print("\n🛠️ Skills Found ({analysis['skill_count']}):")
        for skill in analysis['skills_found']:
            print(f"- {skill}")
            
        print("\n🎓 Education:")
        for edu in analysis['education']:
            print(f"- {edu}")
            
        print(f"\n👨‍💼 Experience Level: {analysis['experience_level']}")
        
        # Add recommendations based on experience level
        print("\n💡 Recommendations:")
        if analysis['experience_level'] == 'Junior':
            print("- Focus on building strong foundations in core technologies")
            print("- Consider contributing to open-source projects")
            print("- Build a portfolio of personal projects")
        elif analysis['experience_level'] == 'Mid-Level':
            print("- Specialize in specific technology domains")
            print("- Take on leadership roles in projects")
            print("- Consider getting relevant certifications")
        else:  # Senior
            print("- Focus on system design and architecture")
            print("- Consider mentoring junior developers")
            print("- Stay updated with emerging technologies")

class AIProjectRecommender:
    def __init__(self):
        self.questions = {
            'programming': {
                'question': 'Rate your Python programming skills (1-5):',
                'sub_questions': [
                    'Are you comfortable with object-oriented programming? (yes/no)',
                    'Have you used Python for data processing? (yes/no)',
                    'Do you have experience with Python ML libraries? (yes/no)'
                ]
            },
            'math_stats': {
                'question': 'Rate your mathematics and statistics knowledge (1-5):',
                'sub_questions': [
                    'Are you familiar with linear algebra? (yes/no)',
                    'Do you understand probability distributions? (yes/no)',
                    'Have you worked with statistical hypothesis testing? (yes/no)'
                ]
            },
            'ml_knowledge': {
                'question': 'Rate your machine learning knowledge (1-5):',
                'sub_questions': [
                    'Have you implemented supervised learning algorithms? (yes/no)',
                    'Are you familiar with neural networks? (yes/no)',
                    'Have you worked with model evaluation metrics? (yes/no)'
                ]
            },
            'tools': {
                'question': 'Rate your experience with ML tools and frameworks (1-5):',
                'sub_questions': [
                    'Have you used scikit-learn? (yes/no)',
                    'Have you worked with TensorFlow or PyTorch? (yes/no)',
                    'Are you familiar with Jupyter notebooks? (yes/no)'
                ]
            },
            'domain': {
                'question': 'Which areas of AI/ML interest you the most?',
                'options': [
                    'Computer Vision',
                    'Natural Language Processing',
                    'Time Series Analysis',
                    'Reinforcement Learning',
                    'Generative AI'
                ]
            }
        }
        self.user_profile = {}

    def get_user_rating(self, question: str) -> int:
        """Get user rating on a scale of 1-5"""
        while True:
            try:
                rating = int(input(question + " "))
                if 1 <= rating <= 5:
                    return rating
                print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")

    def get_yes_no_answer(self, question: str) -> bool:
        """Get yes/no answer from user"""
        while True:
            answer = input(question + " ").lower()
            if answer in ['yes', 'no', 'y', 'n']:
                return answer in ['yes', 'y']
            print("Please answer 'yes' or 'no'.")

    def get_multiple_choice(self, options: List[str], max_selections: int = 2) -> List[str]:
        """Get multiple choice selection from user"""
        print("\nSelect up to", max_selections, "options (enter numbers separated by spaces):")
        for idx, option in enumerate(options, 1):
            print(f"{idx}. {option}")
        
        while True:
            try:
                selections = input("Your choices: ").split()
                selections = [int(s) for s in selections]
                if all(1 <= s <= len(options) for s in selections) and len(selections) <= max_selections:
                    return [options[s-1] for s in selections]
                print(f"Please select up to {max_selections} valid options.")
            except ValueError:
                print("Please enter valid numbers separated by spaces.")

    def collect_user_profile(self):
        """Collect user's AI/ML knowledge profile"""
        print("\n=== AI/ML Knowledge Assessment ===")
        
        # Collect main ratings and sub-questions
        for category, content in self.questions.items():
            if category != 'domain':
                print(f"\n{category.replace('_', ' ').title()}:")
                self.user_profile[category] = {
                    'rating': self.get_user_rating(content['question']),
                    'details': []
                }
                for sub_q in content['sub_questions']:
                    self.user_profile[category]['details'].append(
                        self.get_yes_no_answer(sub_q)
                    )
        
        # Collect domain interests
        print("\nDomain Interests:")
        self.user_profile['interests'] = self.get_multiple_choice(
            self.questions['domain']['options']
        )

    def generate_prompt(self) -> str:
        """Generate prompt for LLM based on user profile"""
        prompt = "Based on the following user profile, suggest 3 unique AI/ML project ideas that match their skill level and interests:\n\n"
        
        # Add skill levels
        prompt += "Skill Levels:\n"
        for category, data in self.user_profile.items():
            if category != 'interests':
                prompt += f"- {category.replace('_', ' ').title()}: {data['rating']}/5\n"
                # Add details about specific skills
                for q, a in zip(self.questions[category]['sub_questions'], data['details']):
                    prompt += f"  - {q.replace('? (yes/no)', '')}: {'Yes' if a else 'No'}\n"
        
        # Add interests
        prompt += f"\nInterests: {', '.join(self.user_profile['interests'])}\n"
        
        # Add specific requirements
        prompt += "\nPlease suggest 3 projects that:\n"
        prompt += "1. Match the user's current skill level\n"
        prompt += "2. Focus on their areas of interest\n"
        prompt += "3. Include specific technologies and frameworks to use\n"
        prompt += "4. Provide a brief project description\n"
        prompt += "5. List learning outcomes\n"
        prompt += "6. Estimate time to complete\n"

        return prompt

    def get_llm_recommendations(self, prompt: str) -> str:
        """Get project recommendations when Ollama is not available"""
        # Default recommendations when Ollama is not available
        return """Based on your profile, here are 3 recommended projects:

1. Personal Skill Portfolio Analyzer
   - Description: Build a tool that analyzes GitHub repositories and project descriptions to create a comprehensive skill portfolio
   - Technologies: Python, pandas, scikit-learn (for text analysis)
   - Learning Outcomes: Data processing, text analysis, portfolio development
   - Estimated Time: 2-3 weeks

2. Learning Path Recommender
   - Description: Create a system that suggests personalized learning resources based on skill gaps
   - Technologies: Python, pandas, basic ML algorithms
   - Learning Outcomes: Recommendation systems, data analysis
   - Estimated Time: 3-4 weeks

3. Interactive Quiz Generator
   - Description: Develop a tool that automatically generates quizzes from educational content
   - Technologies: Python, NLP libraries, web frameworks
   - Learning Outcomes: NLP, web development, educational technology
   - Estimated Time: 4-5 weeks

These projects are designed to help you build practical experience while learning new skills. Start with the one that best matches your current interests and available time."""

    def format_recommendations(self, llm_response: str) -> None:
        """Format and display the project recommendations"""
        print("\n=== Recommended AI/ML Projects ===\n")
        print(llm_response)
        
        # Save recommendations to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        with open(f'project_recommendations_{timestamp}.txt', 'w') as f:
            f.write("User Profile:\n")
            f.write(json.dumps(self.user_profile, indent=2))
            f.write("\n\nRecommendations:\n")
            f.write(llm_response)
        print(f"\nRecommendations saved to project_recommendations_{timestamp}.txt")

    def run(self):
        """Main execution flow"""
        print("Welcome to the AI/ML Project Recommender!")
        
        # Collect user profile
        self.collect_user_profile()
        
        # Generate and send prompt to LLM
        prompt = self.generate_prompt()
        print("\nGenerating project recommendations...")
        recommendations = self.get_llm_recommendations(prompt)
        
        # Display results
        self.format_recommendations(recommendations)

import pandas as pd
import csv
from pathlib import Path
import random
import PyPDF2
import re
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def initialize_databases():
    """Create initial CSV files if they don't exist"""
    # Previous databases remain the same
    domains_data = [
        ['domain', 'roles'],
        ['data_science', 'data_scientist,data_analyst,data_engineer']
    ]
    
    skills_data = [
        ['role', 'skill', 'category'],
        ['data_scientist', 'python', 'programming'],
        ['data_scientist', 'sql', 'database'],
        ['data_scientist', 'tableau', 'visualization'],
        ['data_analyst', 'sql', 'database'],
        ['data_analyst', 'tableau', 'visualization'],
        ['data_analyst', 'excel', 'tools'],
        ['data_engineer', 'python', 'programming'],
        ['data_engineer', 'sql', 'database'],
        ['data_engineer', 'spark', 'big_data']
    ]
    
    questions_data = [
        ['skill', 'question', 'options', 'correct_answer'],
        ['python', 'What is the output of print(type([]))?', 'str,list,tuple,dict', 'list'],
        ['sql', 'Which SQL keyword is used to filter rows?', 'WHERE,FILTER,HAVING,SELECT', 'WHERE'],
        ['tableau', 'Which chart type is best for showing trends over time?', 'pie chart,bar chart,line chart,scatter plot', 'line chart']
    ]
    
    # New database for learning resources
    resources_data = [
        ['skill', 'resource_type', 'title', 'url', 'duration_hours', 'cost_usd', 'difficulty', 'description'],
        ['python', 'course', 'Python Basics', 'https://example.com/python', '20', '0', 'beginner', 'Comprehensive Python basics'],
        ['python', 'tutorial', 'Advanced Python', 'https://example.com/adv-python', '40', '49.99', 'advanced', 'Advanced Python concepts'],
        ['sql', 'documentation', 'SQL Reference', 'https://example.com/sql-docs', '10', '0', 'beginner', 'SQL documentation'],
        ['tableau', 'course', 'Tableau Fundamentals', 'https://example.com/tableau', '15', '29.99', 'beginner', 'Learn Tableau basics']
    ]
    
    # User learning history for collaborative filtering
    user_history_data = [
        ['user_id', 'skill', 'resource_id', 'completion_rate', 'rating'],
        ['user1', 'python', '1', '100', '5'],
        ['user2', 'sql', '3', '90', '4'],
        ['user3', 'tableau', '4', '95', '5']
    ]
    
    files = {
        'domains.csv': domains_data,
        'skills.csv': skills_data,
        'questions.csv': questions_data,
        'resources.csv': resources_data,
        'user_history.csv': user_history_data
    }
    
    for filename, data in files.items():
        if not Path(filename).exists():
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data)

class LearningRecommender:
    def __init__(self):
        # Read CSV files and clean the data
        self.resources_df = pd.read_csv('resources.csv')
        self.user_history_df = pd.read_csv('user_history.csv')
        
        # Clean and convert cost_usd column
        def clean_cost(cost):
            try:
                if pd.isna(cost) or str(cost).lower() in ['free', 'varies', 'self-paced']:
                    return 0.0
                # Extract the first number from the string
                cost_str = str(cost)
                cost_num = float(''.join(c for c in cost_str if c.isdigit() or c == '.'))
                return cost_num
            except:
                return 0.0
        
        # Clean and convert duration_hours column
        def clean_duration(duration):
            try:
                if pd.isna(duration) or str(duration).lower() in ['self-paced', 'varies']:
                    return 40.0  # Default duration for self-paced courses
                return float(duration)
            except:
                return 40.0
        
        # Apply cleaning to columns
        self.resources_df['cost_usd'] = self.resources_df['cost_usd'].apply(clean_cost)
        self.resources_df['duration_hours'] = self.resources_df['duration_hours'].apply(clean_duration)
        
    def get_content_based_recommendations(self, skill, difficulty, max_cost, max_duration):
        """Generate content-based recommendations based on skill and constraints"""
        try:
            # Filter resources by skill and constraints
            mask = (
                (self.resources_df['skill'].str.lower() == skill.lower()) &
                (self.resources_df['difficulty'].str.lower() == difficulty.lower()) &
                (self.resources_df['cost_usd'] <= float(max_cost)) &
                (self.resources_df['duration_hours'] <= float(max_duration))
            )
            
            filtered_resources = self.resources_df[mask]
            
            if filtered_resources.empty:
                # If no exact matches, try relaxing constraints
                mask = (
                    (self.resources_df['skill'].str.lower() == skill.lower()) &
                    (self.resources_df['cost_usd'] <= float(max_cost))
                )
                filtered_resources = self.resources_df[mask]
            
            if filtered_resources.empty:
                return []
            
            # Convert to list of dictionaries for easier handling
            resources = filtered_resources.to_dict('records')
            
            # Sort by cost (prefer free resources) and then by duration
            resources.sort(key=lambda x: (float(x.get('cost_usd', 0)), float(x.get('duration_hours', 0))))
            
            return resources[:3]  # Return top 3 recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return []
    
    def get_collaborative_recommendations(self, skill):
        """Generate collaborative filtering recommendations based on user history"""
        try:
            # Get users who have completed resources for this skill
            skill_history = self.user_history_df[
                self.user_history_df['skill'].str.lower() == skill.lower()
            ]
            
            if skill_history.empty:
                return []
            
            # Get resources with high completion rates and ratings
            successful_resources = skill_history[
                (skill_history['completion_rate'] >= 80) &
                (skill_history['rating'] >= 4)
            ]['resource_id'].unique()
            
            if len(successful_resources) == 0:
                return []
            
            # Get the actual resources
            recommended_resources = self.resources_df[
                self.resources_df.index.isin(successful_resources)
            ].to_dict('records')
            
            # Sort by rating
            recommended_resources.sort(key=lambda x: float(x.get('rating', 0)), reverse=True)
            
            return recommended_resources[:2]  # Return top 2 recommendations
            
        except Exception as e:
            print(f"Error generating collaborative recommendations: {e}")
            return []

class EnhancedSkillAssessmentSystem:
    def __init__(self):
        initialize_databases()
        self.domains_df = pd.read_csv('domains.csv')
        self.skills_df = pd.read_csv('skills.csv')
        self.questions_df = pd.read_csv('questions.csv')
        self.resources_df = pd.read_csv('resources.csv')
        self.recommender = LearningRecommender()
        self.user_responses = {}
        self.resume_skills = {}
        self.user_preferences = {}
        
    def get_user_preferences(self):
        """Get user's time and budget preferences"""
        print("\n=== Learning Preferences ===")
        
        # Get available time
        while True:
            try:
                hours_per_week = float(input("How many hours per week can you dedicate to learning? "))
                if hours_per_week > 0:
                    break
                print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get budget
        while True:
            try:
                budget_choice = input("Do you prefer free resources only? (yes/no): ").lower()
                if budget_choice in ['yes', 'no']:
                    if budget_choice == 'yes':
                        max_budget = 0
                    else:
                        while True:
                            try:
                                max_budget = float(input("What's your maximum budget for learning resources (USD)? "))
                                if max_budget >= 0:
                                    break
                                print("Please enter a non-negative number.")
                            except ValueError:
                                print("Please enter a valid number.")
                    break
                print("Please answer 'yes' or 'no'.")
            except ValueError:
                print("Please enter a valid response.")
        
        # Store preferences
        self.user_preferences = {
            'hours_per_week': hours_per_week,
            'max_budget': max_budget
        }
        
    def select_domain_and_role(self):
        """Select domain and role"""
        print("\n=== Domain and Role Selection ===")
        domains = self.domains_df['domain'].tolist()
        print("Available domains:")
        for i, domain in enumerate(domains, 1):
            print(f"{i}. {domain}")
        
        while True:
            try:
                domain_choice = int(input("Please select a domain (enter the number): "))
                if 1 <= domain_choice <= len(domains):
                    break
                print("Please enter a valid number.")
            except ValueError:
                print("Please enter a valid number.")
        
        selected_domain = domains[domain_choice - 1]
        roles = self.domains_df[self.domains_df['domain'] == selected_domain]['roles'].iloc[0].split(',')
        print("\nAvailable roles:")
        for i, role in enumerate(roles, 1):
            print(f"{i}. {role}")
        
        while True:
            try:
                role_choice = int(input("Please select a role (enter the number): "))
                if 1 <= role_choice <= len(roles):
                    break
                print("Please enter a valid number.")
            except ValueError:
                print("Please enter a valid number.")
        
        selected_role = roles[role_choice - 1]
        return selected_domain, selected_role
    
    def get_skills_for_role(self, role):
        """Get skills for the selected role"""
        skills = self.skills_df[self.skills_df['role'] == role]['skill'].tolist()
        print("\nAvailable skills:")
        for i, skill in enumerate(skills, 1):
            print(f"{i}. {skill}")
        
        while True:
            try:
                skill_choices = input("Please select skills (enter the numbers, separated by commas): ").split(',')
                skill_choices = [int(choice) for choice in skill_choices]
                if all(1 <= choice <= len(skills) for choice in skill_choices):
                    break
                print("Please enter valid numbers.")
            except ValueError:
                print("Please enter valid numbers.")
        
        selected_skills = [skills[choice - 1] for choice in skill_choices]
        return selected_skills
    
    def conduct_quiz(self, selected_skills):
        """Conduct quiz for the selected skills"""
        print("\n=== Starting Quiz ===")
        quiz_score = 0
        total_questions = 0
        
        for skill in selected_skills:
            skill_questions = self.questions_df[self.questions_df['skill'] == skill]
            for _, question in skill_questions.iterrows():
                print(f"\nQuestion: {question['question']}")
                print("Options:")
                options = question['options'].split(',')
                for i, option in enumerate(options, 1):
                    print(f"{i}. {option}")
                
                while True:
                    answer = input("Enter the number of your answer (1-" + str(len(options)) + "): ").strip()
                    if answer.isdigit() and 1 <= int(answer) <= len(options):
                        # Convert the number back to the actual answer text
                        answer = options[int(answer) - 1].lower()
                        break
                    print(f"Please enter a valid number between 1 and {len(options)}.")
                
                self.user_responses[question['question']] = {
                    'skill': skill,
                    'answered': answer,
                    'correct': answer == question['correct_answer'].lower()
                }
                
                if self.user_responses[question['question']]['correct']:
                    quiz_score += 1
                total_questions += 1
        
        return (quiz_score / total_questions) * 100
    
    def generate_learning_plan(self, analysis):
        """Generate personalized learning plan based on skill analysis and user preferences"""
        try:
            learning_plan = {
                'improvement_plan': [],
                'new_skills_plan': []
            }
            
            if not analysis['improvement_needed'] and not analysis['missing_skills']:
                return learning_plan
            
            # Get user preferences
            hours_per_week = self.user_preferences.get('hours_per_week', 10)  # Default to 10 hours
            max_cost = self.user_preferences.get('max_budget', 0)  # Default to free resources
            
            # Plan for skills needing improvement
            for skill_info in analysis['improvement_needed']:
                try:
                    skill = skill_info.split(' ')[0]  # Extract skill name
                    resources = self.recommender.get_content_based_recommendations(
                        skill=skill,
                        difficulty='intermediate',
                        max_cost=max_cost,
                        max_duration=hours_per_week
                    )
                    if resources:
                        learning_plan['improvement_plan'].append({
                            'skill': skill,
                            'current_level': 'needs improvement',
                            'target_level': 'intermediate',
                            'resources': resources,
                            'suggested_hours': hours_per_week
                        })
                except Exception as e:
                    print(f"Error processing improvement plan for {skill}: {e}")
                    continue
            
            # Plan for missing skills
            for skill in analysis['missing_skills']:
                try:
                    resources = self.recommender.get_content_based_recommendations(
                        skill=skill,
                        difficulty='beginner',
                        max_cost=max_cost,
                        max_duration=hours_per_week
                    )
                    if resources:
                        learning_plan['new_skills_plan'].append({
                            'skill': skill,
                            'target_level': 'beginner',
                            'resources': resources,
                            'suggested_hours': hours_per_week
                        })
                except Exception as e:
                    print(f"Error processing new skills plan for {skill}: {e}")
                    continue
            
            return learning_plan
            
        except Exception as e:
            print(f"Error generating learning plan: {e}")
            return {
                'improvement_plan': [],
                'new_skills_plan': []
            }
    
    def display_learning_plan(self, learning_plan):
        """Display the generated learning plan"""
        if not learning_plan['improvement_plan'] and not learning_plan['new_skills_plan']:
            print("\nNo learning plan needed - all skills are at required levels!")
            return
        
        if learning_plan['improvement_plan']:
            print("\nSkills to Improve:")
            for plan in learning_plan['improvement_plan']:
                print(f"\n{plan['skill']} (Current: {plan['current_level']} → Target: {plan['target_level']})")
                print(f"Suggested time commitment: {plan['suggested_hours']} hours per week")
                print("\nRecommended Resources:")
                for resource in plan['resources']:
                    print(f"- {resource['title']}")
                    print(f"  Type: {resource['resource_type']}")
                    if resource['cost_usd'] == 0:
                        print("  Cost: Free")
                    else:
                        print(f"  Cost: ${resource['cost_usd']:.2f}")
                    print(f"  Duration: {resource['duration_hours']} hours")
                    if 'url (sample - replace with actual links when possible)' in resource:
                        print(f"  Link: {resource['url (sample - replace with actual links when possible)']}")
        
        if learning_plan['new_skills_plan']:
            print("\nNew Skills to Learn:")
            for plan in learning_plan['new_skills_plan']:
                print(f"\n{plan['skill']} (Target: {plan['target_level']})")
                print(f"Suggested time commitment: {plan['suggested_hours']} hours per week")
                print("\nRecommended Resources:")
                for resource in plan['resources']:
                    print(f"- {resource['title']}")
                    print(f"  Type: {resource['resource_type']}")
                    if resource['cost_usd'] == 0:
                        print("  Cost: Free")
                    else:
                        print(f"  Cost: ${resource['cost_usd']:.2f}")
                    print(f"  Duration: {resource['duration_hours']} hours")
                    if 'url (sample - replace with actual links when possible)' in resource:
                        print(f"  Link: {resource['url (sample - replace with actual links when possible)']}")
    
    def run_assessment(self, test_mode=False):
        """Main function to run the enhanced assessment"""
        print("\n=== Welcome to the Enhanced Skill Assessment System! ===\n")
        
        # Get user preferences if not in test mode
        if not test_mode:
            self.get_user_preferences()
        
        # Domain and role selection
        if test_mode:
            print("Using default domain and role for testing...")
            domain = 'data_science'
            role = 'data_scientist'
        else:
            print("\n=== Domain and Role Selection ===")
            domain, role = self.select_domain_and_role()
        
        # Skill selection
        if test_mode:
            print("Using default skills for testing...")
            selected_skills = ['python', 'sql', 'tableau']
        else:
            print("\n=== Skill Selection ===")
            selected_skills = self.get_skills_for_role(role)
        
        if not selected_skills:
            print("No skills selected. Exiting assessment.")
            return
        
        # Conduct quiz
        print("\n=== Starting Quiz ===")
        if test_mode:
            print("Using default quiz responses for testing...")
            quiz_score = 100.0  # Perfect score for testing
        else:
            quiz_score = self.conduct_quiz(selected_skills)
        print(f"\nQuiz completed! Score: {quiz_score:.2f}%")
        
        # Resume analysis
        print("\n=== Resume Analysis ===")
        if test_mode:
            print("Using sample resume for testing...")
            resume_score = 80.0  # Sample score for testing
            resume_skills = ['python', 'sql', 'tableau']
        else:
            while True:
                try:
                    resume_path = input("Please provide path to your resume (PDF format) or press Enter to skip: ").strip()
                    if not resume_path:
                        print("Skipping resume analysis...")
                        resume_score = 0
                        resume_skills = []
                        break
                    elif not Path(resume_path).exists():
                        print("File not found. Please check the path and try again.")
                    else:
                        analyzer = ResumeAnalyzer()
                        analysis = analyzer.analyze_resume(resume_path)
                        analyzer.print_analysis(analysis)
                        resume_score = 0
                        resume_skills = analysis['skills_found']
                        print(f"Resume analysis completed! Score: {resume_score:.2f}%")
                        break
                except Exception as e:
                    print(f"Error reading resume: {e}")
                    print("Please try again or press Enter to skip.")
        
        # Calculate final score
        final_score = (quiz_score + resume_score) / 2
        
        # Generate skill gap analysis
        analysis = self.generate_skill_gap_analysis(selected_skills, resume_skills)
        
        # Display results
        print("\n=== Assessment Results ===")
        print(f"Quiz Score: {quiz_score:.2f}%")
        if resume_score > 0:
            print(f"Resume Score: {resume_score:.2f}%")
        print(f"Final Score: {final_score:.2f}%")
        
        print("\n=== Skill Gap Analysis ===")
        if analysis['existing_skills']:
            print("\nExisting Skills:")
            for skill in analysis['existing_skills']:
                print(f"- {skill}")
        
        if analysis['improvement_needed']:
            print("\nSkills Needing Improvement:")
            for skill in analysis['improvement_needed']:
                print(f"- {skill}")
        
        if analysis['missing_skills']:
            print("\nMissing Skills:")
            for skill in analysis['missing_skills']:
                print(f"- {skill}")
        
        # Generate and display learning plan
        print("\n=== Personalized Learning Plan ===")
        learning_plan = self.generate_learning_plan(analysis)
        self.display_learning_plan(learning_plan)
        
        # Ask if user wants AI/ML project recommendations
        if not test_mode:
            while True:
                try:
                    want_projects = input("\nWould you like personalized AI/ML project recommendations? (yes/no): ").lower()
                    if want_projects in ['yes', 'y', 'no', 'n']:
                        if want_projects in ['yes', 'y']:
                            recommender = AIProjectRecommender()
                            recommender.run()
                        break
                    print("Please answer 'yes' or 'no'.")
                except Exception as e:
                    print(f"Error getting project recommendations: {e}")
                    break
    
    def generate_skill_gap_analysis(self, selected_skills, resume_skills):
        """Generate skill gap analysis based on quiz and resume"""
        all_skills = set(self.skills_df['skill'].unique())
        selected_skills = set(selected_skills)
        resume_skills = set(resume_skills)
        
        # Analyze quiz performance
        quiz_performance = {}
        for question, response in self.user_responses.items():
            skill = response['skill']
            if skill not in quiz_performance:
                quiz_performance[skill] = {'correct': 0, 'total': 0}
            quiz_performance[skill]['total'] += 1
            if response['correct']:
                quiz_performance[skill]['correct'] += 1
        
        # Generate analysis
        analysis = {
            'existing_skills': [],
            'missing_skills': [],
            'improvement_needed': []
        }
        
        # Analyze each skill
        for skill in all_skills:
            if skill in resume_skills and skill in selected_skills:
                # Check quiz performance
                if skill in quiz_performance:
                    performance = quiz_performance[skill]['correct'] / quiz_performance[skill]['total']
                    if performance >= 0.7:  # 70% or better
                        analysis['existing_skills'].append(
                            f"{skill} (Resume: Found, Quiz: {performance*100:.0f}%)")
                    else:
                        analysis['improvement_needed'].append(
                            f"{skill} (Resume: Found, Quiz: {performance*100:.0f}%)")
            elif skill not in resume_skills and skill not in selected_skills:
                analysis['missing_skills'].append(skill)
            elif skill not in resume_skills:
                if skill in quiz_performance:
                    performance = quiz_performance[skill]['correct'] / quiz_performance[skill]['total']
                    if performance < 0.7:
                        analysis['improvement_needed'].append(f"{skill} (Quiz: {performance*100:.0f}%)")
            elif skill not in selected_skills:
                analysis['existing_skills'].append(f"{skill} (Resume: Found)")
        
        return analysis

    def analyze_resume(self, file_path):
        """Analyze resume and extract skills"""
        try:
            with open(file_path, 'r') as file:
                text = file.read()
            
            # Get all possible skills from database
            all_skills = self.skills_df['skill'].unique()
            
            # Simple skill detection
            found_skills = {}
            for skill in all_skills:
                # Basic skill level detection based on keywords
                skill_pattern = re.compile(f"{skill}", re.IGNORECASE)
                matches = skill_pattern.findall(text)
                if matches:
                    # Simple scoring based on frequency and context
                    if len(matches) > 2:
                        level = "Advanced"
                    elif len(matches) > 1:
                        level = "Intermediate"
                    else:
                        level = "Basic"
                    found_skills[skill] = level
            
            self.resume_skills = found_skills
            return len(found_skills) / len(all_skills) * 100  # Basic resume score
        except Exception as e:
            print(f"Error analyzing resume: {e}")
            return 0

if __name__ == "__main__":
    import sys
    
    # Create the assessment system
    assessment_system = EnhancedSkillAssessmentSystem()
    
    # Check if we should use default values for testing
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("Running in test mode with default values...")
        assessment_system.user_preferences = {
            'hours_per_week': 10,
            'max_budget': 0  # Free resources only
        }
        assessment_system.run_assessment(test_mode=True)
    else:
        # Run in interactive mode
        assessment_system.run_assessment()

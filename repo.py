import requests

def get_repos_from_org(org_name, token):
    repo_details = []
    page = 1
    per_page = 100  # Maximum allowed by GitHub API

    while True:
        url = f"https://api.github.com/orgs/{org_name}/repos"
        headers = {'Authorization': f'token {token}'}
        params = {'page': page, 'per_page': per_page}

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            repos = response.json()

            if not repos:
                break

            for repo in repos:
                repo_info = {
                    'name': repo['name'],
                    'full_name': repo['full_name'],
                    'description': repo['description'],
                    'html_url': repo['html_url'],
                    'created_at': repo['created_at'],
                    'updated_at': repo['updated_at'],
                    'language': repo['language'],
                    'stargazers_count': repo['stargazers_count'],
                    'forks_count': repo['forks_count'],
                    'open_issues_count': repo['open_issues_count']
                }
                repo_details.append(repo_info)

            page += 1

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            break

    return repo_details

def write_to_markdown(repos, filename='huggingface.md'):
    with open(filename, 'w', encoding="UTF-8") as f:
        f.write("# Repository Details\n\n")
        for repo in repos:
            f.write(f"## {repo['name']}\n")
            f.write(f"- **Full Name**: {repo['full_name']}\n")
            f.write(f"- **Description**: {repo['description']}\n")
            f.write(f"- **URL**: [Link]({repo['html_url']})\n")
            f.write(f"- **Created At**: {repo['created_at']}\n")
            f.write(f"- **Updated At**: {repo['updated_at']}\n")
            f.write(f"- **Language**: {repo['language']}\n")
            f.write(f"- **Stars**: {repo['stargazers_count']}\n")
            f.write(f"- **Forks**: {repo['forks_count']}\n")
            f.write(f"- **Open Issues**: {repo['open_issues_count']}\n\n")



# Example Usage
GITHUB_TOKEN = 'XXXXXX'  # Replace with your GitHub personal access token
organization_name = 'huggingface'
repos = get_repos_from_org(organization_name, GITHUB_TOKEN)

write_to_markdown(repos)

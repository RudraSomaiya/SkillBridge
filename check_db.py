from app import create_app, db
from app.models.domain import Domain, Role, Skill

app = create_app()

with app.app_context():
    print("\nDomains:")
    domains = Domain.query.all()
    for domain in domains:
        print(f"- {domain.name} (ID: {domain.id})")
        
    print("\nRoles:")
    roles = Role.query.all()
    for role in roles:
        print(f"- {role.name} (ID: {role.id}, Domain ID: {role.domain_id})")
        
    print("\nSkills:")
    skills = Skill.query.all()
    for skill in skills:
        print(f"- {skill.name} (ID: {skill.id}, Role ID: {skill.role_id})")

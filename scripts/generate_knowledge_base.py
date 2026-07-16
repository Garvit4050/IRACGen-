#!/usr/bin/env python3
"""
Legal Knowledge Base Generator
Creates comprehensive set of landmark legal cases for LegalRAG system
"""

import os
from pathlib import Path

# Employment Law Cases
EMPLOYMENT_CASES = [
    {
        "name": "McDonnell Douglas Corp. v. Green",
        "citation": "411 U.S. 792 (1973)",
        "court": "Supreme Court of the United States",
        "facts": "Percy Green, an African American, was laid off by McDonnell Douglas as part of a workforce reduction. Green engaged in protests against the company, including blocking access to the plant. Later, McDonnell Douglas advertised for mechanics, and Green applied but was not rehired. Green claimed race discrimination and retaliation.",
        "issue": "What is the framework for analyzing employment discrimination claims under Title VII?",
        "holding": "The Court established a three-part burden-shifting framework for discrimination cases.",
        "reasoning": "First, plaintiff must establish prima facie case of discrimination. Second, employer must articulate legitimate, non-discriminatory reason for action. Third, plaintiff must prove the reason is pretextual.",
        "principles": [
            "McDonnell Douglas burden-shifting framework",
            "Prima facie case requires: (1) protected class, (2) qualified, (3) adverse action, (4) circumstances suggesting discrimination",
            "Employer's burden of production, not persuasion",
            "Plaintiff retains ultimate burden of proving intentional discrimination"
        ]
    },
    {
        "name": "Garcetti v. Ceballos",
        "citation": "547 U.S. 410 (2006)",
        "court": "Supreme Court of the United States",
        "facts": "Richard Ceballos, a deputy district attorney, was allegedly subject to retaliatory employment actions after he wrote a memo questioning the veracity of an affidavit used to obtain a search warrant. He claimed First Amendment protection for his memo.",
        "issue": "Do public employees have First Amendment protection for speech made pursuant to their official duties?",
        "holding": "Public employees do not have First Amendment protection for speech made pursuant to their official job duties.",
        "reasoning": "When public employees make statements pursuant to their official duties, they are not speaking as citizens for First Amendment purposes. The government employer needs broad discretion to manage its operations.",
        "principles": [
            "Public employee speech test: speaking as citizen on matter of public concern",
            "Official duty speech not protected",
            "Pickering balancing test applies only when employee speaks as citizen",
            "Government has greater power to control employee speech than private citizen speech"
        ]
    },
    {
        "name": "Price Waterhouse v. Hopkins",
        "citation": "490 U.S. 228 (1989)",
        "court": "Supreme Court of the United States",
        "facts": "Ann Hopkins was denied partnership at Price Waterhouse despite strong performance. Comments about her allegedly insufficiently feminine behavior were made by partners. Some suggested she take a course at charm school.",
        "issue": "In a Title VII case, when discrimination is shown to be a motivating factor, what is the employer's burden?",
        "holding": "Once plaintiff shows discrimination was a motivating factor, employer must prove it would have made the same decision absent discrimination.",
        "reasoning": "Title VII meant to strike at entire spectrum of disparate treatment. When illegitimate factor plays a motivating part, employer can avoid liability only by proving it would have made same decision without that factor.",
        "principles": [
            "Mixed-motive discrimination theory",
            "Motivating factor test (later codified in 1991 Civil Rights Act)",
            "Employer's same-decision defense",
            "Sex stereotyping is form of sex discrimination"
        ]
    }
]

# Contract Law Cases
CONTRACT_CASES = [
    {
        "name": "Hadley v. Baxendale",
        "citation": "9 Ex. 341 (1854)",
        "court": "Court of Exchequer (England)",
        "facts": "Hadley's mill was stopped by broken crankshaft. Hadley hired Baxendale to transport broken shaft to manufacturer to make replacement. Baxendale delayed delivery. Hadley sued for lost profits during delay.",
        "issue": "What damages are recoverable for breach of contract?",
        "holding": "Damages must be reasonably foreseeable as probable result of breach at time contract was made.",
        "reasoning": "Damages should be such as may fairly and reasonably be considered arising naturally from breach, or such as may reasonably be supposed to have been in contemplation of both parties at contract formation.",
        "principles": [
            "Foreseeability rule for contract damages",
            "Two types: (1) arising naturally, (2) in parties' contemplation",
            "Special circumstances must be communicated",
            "Limits recoverable consequential damages"
        ]
    },
    {
        "name": "Lucy v. Zehmer",
        "citation": "196 Va. 493 (1954)",
        "court": "Supreme Court of Virginia",
        "facts": "Zehmer wrote agreement to sell farm to Lucy for $50,000 while drinking at restaurant. Zehmer claimed it was a joke. Lucy sought specific performance.",
        "issue": "Is a contract formed when one party claims he was joking?",
        "holding": "Contract is formed if reasonable person would believe offer was serious, regardless of secret subjective intent.",
        "reasoning": "Mental assent of parties is not requisite for contract formation. If words and acts, judged by reasonable standard, manifest intention to agree, it is immaterial what may be real but unexpressed state of mind.",
        "principles": [
            "Objective theory of contracts",
            "Reasonable person standard for contract formation",
            "Secret intent irrelevant",
            "Outward expression controls over inward intention"
        ]
    },
    {
        "name": "Hamer v. Sidway",
        "citation": "124 N.Y. 538 (1891)",
        "court": "New York Court of Appeals",
        "facts": "Uncle promised nephew $5,000 if nephew refrained from drinking, smoking, swearing, and gambling until age 21. Nephew complied. Uncle died before paying. Estate claimed no consideration.",
        "issue": "Does forbearance of a legal right constitute valid consideration?",
        "holding": "Forbearance of legal right, even if detrimental only to promisee, is sufficient consideration.",
        "reasoning": "Consideration means that promisor receives something of value or promisee gives up legal right. Nephew gave up legal rights to drink, smoke, etc. This is sufficient consideration even if arguably beneficial to nephew.",
        "principles": [
            "Forbearance as consideration",
            "Detriment to promisee sufficient",
            "Benefit to promisor not required",
            "Legal right waiver creates value"
        ]
    }
]

# Tort Law Cases
TORT_CASES = [
    {
        "name": "Palsgraf v. Long Island Railroad Co.",
        "citation": "248 N.Y. 339 (1928)",
        "court": "New York Court of Appeals",
        "facts": "Railroad guards helped passenger board moving train. Package containing fireworks fell and exploded. Shock waves knocked over scales far away, injuring Palsgraf. She sued railroad.",
        "issue": "To whom does a defendant owe a duty of care?",
        "holding": "Duty of care is owed only to foreseeable plaintiffs within the zone of danger.",
        "reasoning": "Negligence is not actionable unless it involves violation of a legally protected interest. There must be duty to the injured party. Defendant's conduct was not negligent as to Palsgraf because she was not foreseeable plaintiff.",
        "principles": [
            "Duty limited to foreseeable plaintiffs",
            "Zone of danger test",
            "No duty to unforeseeable plaintiffs",
            "Proximate cause requires foreseeability"
        ]
    },
    {
        "name": "MacPherson v. Buick Motor Co.",
        "citation": "217 N.Y. 382 (1916)",
        "court": "New York Court of Appeals",
        "facts": "MacPherson bought Buick from dealer. Defective wheel collapsed, injuring MacPherson. Buick bought wheel from another manufacturer. MacPherson sued Buick directly.",
        "issue": "Can manufacturer be liable to ultimate consumer with whom it has no contract (no privity)?",
        "holding": "Manufacturer owes duty of care to foreseeable users of dangerous products, regardless of privity of contract.",
        "reasoning": "If product is reasonably certain to be dangerous if negligently made, manufacturer owes duty to all foreseeable users. Privity of contract not required when danger is foreseeable.",
        "principles": [
            "Abolition of privity requirement in products liability",
            "Duty to foreseeable users",
            "Inherently dangerous products",
            "Foundation of modern products liability"
        ]
    },
    {
        "name": "Rylands v. Fletcher",
        "citation": "L.R. 3 H.L. 330 (1868)",
        "court": "House of Lords (England)",
        "facts": "Rylands built reservoir on his land. Water escaped through old mine shafts and flooded Fletcher's mine. Accident, not negligence. Fletcher sued.",
        "issue": "Is there strict liability for dangerous activities?",
        "holding": "Person who brings something dangerous onto land that escapes and causes damage is strictly liable.",
        "reasoning": "Person who for his own purposes brings on lands something likely to do mischief if it escapes, must keep it at his peril. If he does not, he is answerable for all damage which is natural consequence of its escape.",
        "principles": [
            "Strict liability for abnormally dangerous activities",
            "Non-natural use of land",
            "No fault required",
            "Foundation of strict liability doctrine"
        ]
    }
]

def format_case(case):
    """Format case into structured text"""
    principles_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(case['principles'])])
    
    return f"""Case Name: {case['name']}
Citation: {case['citation']}
Court: {case['court']}

FACTS:
{case['facts']}

ISSUE:
{case['issue']}

HOLDING:
{case['holding']}

REASONING:
{case['reasoning']}

LEGAL PRINCIPLES:
{principles_text}

SIGNIFICANCE:
This case is a landmark precedent in its area of law and is regularly cited in legal analysis and court decisions.
"""

def generate_knowledge_base():
    """Generate all legal case files"""
    base_dir = Path("data/legal_cases")
    
    # Employment Law
    emp_dir = base_dir / "employment_law"
    emp_dir.mkdir(parents=True, exist_ok=True)
    
    for case in EMPLOYMENT_CASES:
        filename = case['name'].lower().replace(' ', '_').replace('.', '').replace(',', '') + '.txt'
        filepath = emp_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(format_case(case))
        print(f"✓ Created: {filepath}")
    
    # Contract Law
    contract_dir = base_dir / "contract_law"
    contract_dir.mkdir(parents=True, exist_ok=True)
    
    for case in CONTRACT_CASES:
        filename = case['name'].lower().replace(' ', '_').replace('.', '').replace(',', '') + '.txt'
        filepath = contract_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(format_case(case))
        print(f"✓ Created: {filepath}")
    
    # Tort Law
    tort_dir = base_dir / "tort_law"
    tort_dir.mkdir(parents=True, exist_ok=True)
    
    for case in TORT_CASES:
        filename = case['name'].lower().replace(' ', '_').replace('.', '').replace(',', '') + '.txt'
        filepath = tort_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(format_case(case))
        print(f"✓ Created: {filepath}")
    
    # Create index file
    create_index(base_dir)
    
    print(f"\n✅ Legal knowledge base created!")
    print(f"   Employment Law: {len(EMPLOYMENT_CASES)} cases")
    print(f"   Contract Law: {len(CONTRACT_CASES)} cases")
    print(f"   Tort Law: {len(TORT_CASES)} cases")
    print(f"   Total: {len(EMPLOYMENT_CASES) + len(CONTRACT_CASES) + len(TORT_CASES)} cases")

def create_index(base_dir):
    """Create index of all cases"""
    index_content = """# Legal Knowledge Base Index

## Employment Law Cases
1. Griggs v. Duke Power Co., 401 U.S. 424 (1971) - Disparate Impact
2. McDonnell Douglas Corp. v. Green, 411 U.S. 792 (1973) - Burden-Shifting Framework
3. Garcetti v. Ceballos, 547 U.S. 410 (2006) - Public Employee Speech
4. Price Waterhouse v. Hopkins, 490 U.S. 228 (1989) - Mixed-Motive Discrimination

## Contract Law Cases
1. Hadley v. Baxendale, 9 Ex. 341 (1854) - Contract Damages (Foreseeability)
2. Lucy v. Zehmer, 196 Va. 493 (1954) - Contract Formation (Objective Theory)
3. Hamer v. Sidway, 124 N.Y. 538 (1891) - Consideration (Forbearance)

## Tort Law Cases
1. Palsgraf v. Long Island Railroad Co., 248 N.Y. 339 (1928) - Duty of Care
2. MacPherson v. Buick Motor Co., 217 N.Y. 382 (1916) - Products Liability
3. Rylands v. Fletcher, L.R. 3 H.L. 330 (1868) - Strict Liability

## Sources
All cases are landmark precedents from U.S. Supreme Court, state supreme courts, or English common law.
All cases are public domain and freely accessible via Google Scholar Case Law.

## How to Add More Cases
1. Find case on Google Scholar (scholar.google.com)
2. Copy full text
3. Format using the template in generate_knowledge_base.py
4. Add to appropriate category directory
5. Update this index
"""
    
    index_path = base_dir / "INDEX.md"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    print(f"✓ Created: {index_path}")

if __name__ == "__main__":
    generate_knowledge_base()

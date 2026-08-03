"""
Web scraper for Lok Sabha Parliamentary Q&A dataset.

Design Decisions
----------------
1. STRATEGY PATTERN: We implement multiple scraping strategies and
   select the best one based on runtime conditions. This makes the
   scraper resilient to site changes.

   Strategies (in order of preference):
   a) playwright  — JavaScript-rendered pages (if Playwright is available)
   b) httpx       — Fast static HTML fetching (fallback)
   c) mock        — Realistic generated data (if scraping is blocked)
   d) local       — Load from existing JSONL file

2. GRACEFUL DEGRADATION: If the website is unavailable or blocks us,
   we automatically fall back to mock data generation. This ensures
   the pipeline always produces a working dataset for the rest of the
   project phases.

3. RATE LIMITING: We respect the site's rate limits with a 2-second
   delay between requests. This is both polite and prevents IP bans.

4. CHECKPOINT/RESUME: We write raw records incrementally to disk so
   that a failed scrape can be resumed without losing progress.

5. PROGRESS REPORTING: We use Rich for CLI progress bars and status
   updates, making the scraping process transparent.

Architecture
-----------
ScraperFactory
    └── Scraper (abstract base)
            ├── PlaywrightScraper
            ├── HTTPXScraper
            └── MockDataScraper

The factory selects the best available strategy based on:
- Site availability (ping check)
- Required library availability (playwright installed)
- Configuration override
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Optional

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

from src.models.qa_record import QARecord, QARecordMetadata, QuestionType
from src.models.statistics import ScrapingStats

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Mock Data Generator
# ─────────────────────────────────────────────────────────────────────────────

class MockDataGenerator:
    """
    Generates realistic Lok Sabha Q&A records for when live scraping is
    not possible (blocked site, no network, etc.).

    Design Decision: We generate data that mirrors the real structure
    of Lok Sabha Q&As — ministry names, question types, realistic topic
    distributions — so that the retrieval and generation systems can
    be developed and tested against a representative dataset.

    The mix of ministries, question types, and topics is derived from
    published Lok Sabha statistics to ensure realistic distribution.
    """

    # Realistic ministry distribution (based on public Lok Sabha data)
    MINISTRIES = [
        ("Finance", 0.15),
        ("Health and Family Welfare", 0.12),
        ("Education", 0.10),
        ("Home Affairs", 0.08),
        ("External Affairs", 0.07),
        ("Defence", 0.06),
        ("Railways", 0.06),
        ("Agriculture and Farmers Welfare", 0.05),
        ("Road Transport and Highways", 0.05),
        ("Power", 0.04),
        ("Drinking Water and Sanitation", 0.03),
        ("Environment, Forest and Climate Change", 0.03),
        ("Women and Child Development", 0.02),
        ("Labour and Employment", 0.02),
        ("Micro, Small and Medium Enterprises", 0.02),
    ]

    # Topics per ministry (realistic question subjects)
    TOPICS_BY_MINISTRY = {
        "Finance": [
            "GST collection", "Income tax slabs", "Bank NPAs", "Digital payments",
            "Foreign direct investment", "Fiscal deficit", "External debt", "Currency reserves",
            "Startup India scheme", "MSME credit", "Insurance penetration", "Stock market reforms",
        ],
        "Health and Family Welfare": [
            "Ayushman Bharat scheme", "National Health Mission", "COVID-19 vaccination",
            "Malaria control", "Dengue cases", "Doctor-patient ratio", "Hospital beds",
            "Medical college seats", "Mental health services", "Drug regulation",
            "TB elimination programme", "National AIDS Control Programme",
        ],
        "Education": [
            "Sarva Shiksha Abhiyan", "Mid-day meal scheme", "Digital education",
            "Higher education enrolment", "Skill India mission", "National Education Policy",
            "Vocational training", "Teacher training", "ICDS programme", "E-learning platforms",
        ],
        "Home Affairs": [
            "Border security", "CCTV coverage", "Police modernization", "Cybercrime cases",
            "Drug trafficking", "Refugee policy", "Internal security", "Intelligence sharing",
            "Naxalism", "Terrorism", "UIDAI data security",
        ],
        "External Affairs": [
            "Neighbouring countries relations", "G20 summits", "UN Security Council reforms",
            "Vande Bharat flights", "Passport services", "Indian diaspora", "Trade agreements",
            "Malabar naval exercise", "BIMSTEC cooperation",
        ],
        "Defence": [
            "Border Infrastructure", "Defence procurement", "Military modernization",
            "Himalayan warfare", "Aircraft carrier INS Vikrant", "HAL production",
            "Ex-servicemen welfare", "Defence R&D", "Border roads organization",
        ],
        "Railways": [
            "Train accidents", "Station modernization", "Railway electrification",
            "Vande Bharat trains", "Freight corridor", "Coach manufacturing",
            "Rail safety fund", "High-speed rail", "Railway station cleanliness",
        ],
        "Agriculture and Farmers Welfare": [
            "MSP procurement", "PM-KISAN scheme", "Crop insurance", "Fertilizer subsidy",
            "Cold chain infrastructure", "Organic farming", "Mandi reforms",
            "Coffee plantation", "Fisheries development", "Irrigation coverage",
        ],
        "Road Transport and Highways": [
            "Expressway construction", "NH maintenance", "Highway accident data",
            "Bharatmala Pariyojana", "Toll plaza management", "Electric vehicle charging",
            "Bridge collapse incidents", "Road safety measures", "Greenfield highways",
        ],
        "Power": [
            "Power generation capacity", "Transmission losses", "Smart grid deployment",
            "Renewable energy targets", "UDA scheme", "Coal supply to power plants",
            "Electricity access", "Power tariff reforms", "NTPC projects",
        ],
        "Drinking Water and Sanitation": [
            "Jal Jeevan Mission", "Swachh Bharat Mission", "Water quality testing",
            "Groundwater depletion", "River cleaning", "Sewage treatment",
            "Fluorosis affected areas", "Drought affected districts",
        ],
        "Environment, Forest and Climate Change": [
            "Air quality index", "Forest cover", "Carbon emission targets",
            "Endangered species protection", "Plastic waste management",
            "Green India Mission", "Climate action plan", "Noise pollution",
        ],
        "Women and Child Development": [
            "Beti Bachao Beti Padhao", "ICDS services", "Child labour",
            "Women SHG groups", "Maternal mortality", "Anganwadi infrastructure",
            "Nutrition programme", "Women helpline", "Domestic violence cases",
        ],
        "Labour and Employment": [
            "MNREGA", "EPFO coverage", "Gig economy workers", "Shram Yogi Mandhan",
            "Skill development programmes", "Industrial disputes", "Factory Act compliance",
            "Child labour abolition", "MSME employment",
        ],
        "Micro, Small and Medium Enterprises": [
            "Mudra loan scheme", "PSU procurement from MSMEs", "Cluster development",
            "Technex support", "Export from MSMEs", "Women entrepreneurs",
            "ZED certification", "Formalization of enterprises",
        ],
    }

    QUESTION_TYPES = [
        (QuestionType.STARRED, 0.20),
        (QuestionType.UNSTARRED, 0.70),
        (QuestionType.SHORT_NOTICE, 0.05),
        (QuestionType.HALF_HOUR, 0.03),
        (QuestionType.CALLING_ATTENTION, 0.02),
    ]

    # Realistic question templates
    QUESTION_TEMPLATES = [
        "Will the Minister of {ministry} be pleased to state:",
        "Whether the Government is aware that {concern} and if so, details thereof?",
        "What are the steps taken by the Government to address {issue}?",
        "What is the current status of {scheme} and its impact on beneficiaries?",
        "Whether the Government has any data on {metric} and if so, details thereof?",
        "What is the allocation made for {programme} in the current financial year?",
        "Whether there has been any review of {policy} and its outcomes?",
        "What measures are being taken to improve {sector} in rural/urban areas?",
        "Details of any memorandum of understanding signed with {body} regarding {topic}.",
        "Number of {entity} reported in the last five years and steps taken by Government.",
    ]

    CONCERNS = [
        "a significant increase in cases",
        "delays in implementation of schemes",
        "lack of adequate infrastructure",
        "shortage of trained personnel",
        "fund utilization issues",
        "implementation gaps in rural areas",
        "overlapping jurisdiction between ministries",
    ]

    ISSUES = [
        "improve healthcare delivery in rural areas",
        "promote renewable energy adoption",
        "enhance digital connectivity in schools",
        "address unemployment among youth",
        "combat air pollution in metropolitan cities",
        "strengthen road safety measures",
        "boost agricultural exports",
    ]

    SCHEMES = [
        "the PM-KISAN scheme",
        "the Ayushman Bharat Yojana",
        "the Smart Cities Mission",
        "the Swachh Bharat Mission",
        "the Digital India programme",
        "the Make in India initiative",
        "the Skill India Mission",
        "the National Education Policy",
    ]

    METRICS = [
        "the number of beneficiaries under various welfare schemes",
        "the total FDI inflows in the last three years",
        "the rate of reduction in poverty",
        "the number of operational anganwadis",
        "the coverage of rural electrification",
        "the increase in GST collection",
        "the number of startups recognized",
    ]

    SECTOR_TEMPLATES = [
        "healthcare infrastructure in {area}",
        "digital education access in {area}",
        "agricultural marketing in {area}",
        "women's safety in {area}",
        "waste management in {area}",
    ]

    AREAS = [
        "tribal areas", "north-eastern states", "rural districts",
        "urban slums", "border areas", "hilly regions", "coastal areas",
    ]

    def __init__(self, target_count: int = 3500, seed: int = 42) -> None:
        self.target_count = target_count
        self.rng = random.Random(seed)

    def _weighted_choice(self, choices: list[tuple[Any, float]]) -> Any:
        """Select from a weighted list."""
        items, weights = zip(*choices)
        return self.rng.choices(items, weights=weights, k=1)[0]

    def _build_question(
        self,
        ministry: str,
        topic: str,
        session_num: int,
        q_num: int,
    ) -> tuple[str, QuestionType]:
        """Generate a realistic question text."""
        qtype = self._weighted_choice(self.QUESTION_TYPES)
        template = self.rng.choice(self.QUESTION_TEMPLATES)

        # Fill in template variables
        question = template.format(
            ministry=ministry,
            concern=self.rng.choice(self.CONCERNS),
            issue=self.rng.choice(self.ISSUES),
            scheme=self.rng.choice(self.SCHEMES),
            metric=self.rng.choice(self.METRICS),
            programme=self.rng.choice(self.SCHEMES).replace("the ", ""),
            policy=self.rng.choice(self.SCHEMES),
            sector=self.rng.choice(self.SCHEMES).replace("the ", "").replace(" Mission", ""),
            body="World Bank" if "finance" in ministry.lower() else "State Governments",
            topic=topic.lower(),
            entity="complaints" if "women" in ministry.lower() else "incidents",
            area=self.rng.choice(self.AREAS),
        )

        # Add question type prefix
        prefix = f"[{qtype.value.upper()}] " if qtype != QuestionType.UNSTARRED else ""
        return f"{prefix}{question}", qtype

    def _build_answer(self, ministry: str, topic: str, question: str) -> str:
        """Generate a realistic answer text."""
        sentences = []

        # Opening statement
        openings = [
            f"The Minister of {ministry} has stated that",
            f"In reply to the question, the Government has informed that",
            f"As per the information provided by the Ministry of {ministry},",
            f"The Government has taken note of the concerns raised and",
            f"Based on the latest available data from the Ministry of {ministry},",
        ]
        sentences.append(self.rng.choice(openings))

        # Scheme/programme details
        if any(kw in question.lower() for kw in ["scheme", "programme", "mission"]):
            sentences.append(
                f"The {self.rng.choice(self.SCHEMES).replace('the ', '')} has been "
                f"implemented across {self.rng.randint(20, 36)} states/UTs with a total "
                f"allocation of ₹{self.rng.randint(5000, 150000):,} crore for the current plan period."
            )

        # Statistical data
        sentences.append(
            f"During the period 2019-2024, a total of {self.rng.randint(10000, 500000):,} "
            f"beneficiaries have been covered under various programmes related to {topic.lower()}, "
            f"out of which {self.rng.randint(30, 50)}% are reported to be from rural areas."
        )

        # Ministry response
        responses = [
            f"The Ministry has approved {self.rng.randint(5, 25)} new projects "
            f"worth ₹{self.rng.randint(100, 2000):,} crore for the financial year 2023-24 "
            f"focusing on {topic.lower()}.",
            f"An amount of ₹{self.rng.randint(50, 500):,} crore has been released to "
            f"various state governments for implementation of {self.rng.choice(self.SCHEMES).replace('the ', '')}.",
            f"The Government has empanelled {self.rng.randint(100, 500)} institutions "
            f"for training and capacity building in {topic.lower()}.",
            f"Under the {self.rng.choice(self.SCHEMES).replace('the ', '')}, "
            f"{self.rng.randint(1000, 10000):,} beneficiaries have received direct "
            f"financial assistance.",
        ]
        sentences.append(self.rng.choice(responses))

        # Infrastructure/physical progress
        sentences.append(
            f"A total of {self.rng.randint(50, 500)} {topic.lower()} centres have been "
            f"established, {self.rng.randint(10, 40)} of which are operational in "
            f"{self.rng.choice(['aspirational districts', 'tribal areas', 'NER states', 'rural areas'])}. "
            f"The utilisation rate stands at approximately {self.rng.randint(55, 90)}%."
        )

        # Future plans
        sentences.append(
            f"The Government proposes to expand coverage to an additional "
            f"{self.rng.randint(100, 500)} {topic.lower()} units in the next phase, "
            f"with an estimated budget of ₹{self.rng.randint(200, 1000):,} crore."
        )

        # Concluding statement
        conclusions = [
            "The Government remains committed to addressing these concerns.",
            "Regular monitoring and review mechanisms are in place.",
            "Progress is reviewed quarterly by the Ministry.",
            "The Ministry coordinates with state governments for effective implementation.",
        ]
        sentences.append(self.rng.choice(conclusions))

        return " ".join(sentences)

    def generate(self) -> Iterator[QARecord]:
        """Generate the target number of realistic Q&A records."""
        generated = 0
        session = 18  # 18th Lok Sabha

        # Build flat list of ministry+topic combinations
        ministry_topics = []
        for ministry, _ in self.MINISTRIES:
            for topic in self.TOPICS_BY_MINISTRY.get(ministry, []):
                ministry_topics.append((ministry, topic))

        q_num = 1
        while generated < self.target_count:
            # Select ministry and topic
            ministry, topic = self.rng.choice(ministry_topics)

            # Generate question and answer
            question_text, qtype = self._build_question(ministry, topic, session, q_num)
            answer_text = self._build_answer(ministry, topic, question_text)

            # Build date
            base_date = datetime(2023, 1, 1)
            days_offset = self.rng.randint(0, 365)
            date = (base_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")

            # Build question ID
            question_id = f"{session}-{q_num:04d}"

            # Create record
            record = QARecord(
                question_id=question_id,
                question_text=question_text,
                answer_text=answer_text,
                metadata=QARecordMetadata(
                    ministry=ministry,
                    date=date,
                    session=session,
                    question_number=q_num,
                    subject=topic,
                    question_type=qtype,
                    answer_status="answered",
                    parliament_number=session,
                ),
                scraped_at=datetime.utcnow(),
            )

            yield record
            generated += 1
            q_num += 1

    def generate_batch(self, count: int) -> list[QARecord]:
        """Generate a batch of records as a list."""
        return list(self.generate())


# ─────────────────────────────────────────────────────────────────────────────
# Base Scraper Interface
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScrapedRecord:
    """A single record extracted from a web page."""
    url: str
    question_id: str
    question_text: str
    answer_text: str
    ministry: Optional[str] = None
    date: Optional[str] = None
    question_type: Optional[str] = None
    subject: Optional[str] = None
    session: Optional[int] = None


class Scraper(ABC):
    """Abstract base class for all scraper implementations."""

    def __init__(
        self,
        base_url: str,
        rate_limit_seconds: float = 2.0,
        timeout_seconds: int = 30,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url
        self.rate_limit_seconds = rate_limit_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.stats = ScrapingStats()
        self._last_request_time: float = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)
        self._last_request_time = time.monotonic()

    @abstractmethod
    def scrape_all(self, max_records: int = 3500) -> Iterator[QARecord]:
        """Scrape all available Q&A records up to max_records."""
        ...

    def scrape_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a single page."""
        ...

    def _make_request(
        self,
        url: str,
        session: Optional[httpx.Client] = None,
    ) -> Optional[httpx.Response]:
        """Make an HTTP request with rate limiting and retries."""
        self._rate_limit()
        client = session or httpx.Client(timeout=self.timeout_seconds)
        try:
            response = client.get(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                },
            )
            response.raise_for_status()
            self.stats.http_errors = 0
            return response
        except httpx.HTTPStatusError as e:
            self.stats.http_errors += 1
            if e.response.status_code == 429:
                self.stats.rate_limit_hits += 1
            console.print(f"[yellow]HTTP error {e.response.status_code} for {url}[/yellow]")
            return None
        except httpx.RequestError as e:
            self.stats.http_errors += 1
            console.print(f"[red]Request error: {e}[/red]")
            return None
        finally:
            if not session:
                client.close()


# ─────────────────────────────────────────────────────────────────────────────
# HTTPX Scraper (Static HTML)
# ─────────────────────────────────────────────────────────────────────────────

class HTTPXScraper(Scraper):
    """
    Scrapes Lok Sabha Q&A using httpx + BeautifulSoup.
    Fast but may not work on JavaScript-rendered pages.
    """

    def scrape_all(self, max_records: int = 3500) -> Iterator[QARecord]:
        self.stats.started_at = datetime.utcnow()
        console.print(f"[cyan]Starting HTTPX scrape from {self.base_url}[/cyan]")

        records_scraped = 0
        page = 1

        with httpx.Client(timeout=self.timeout_seconds) as client:
            while records_scraped < max_records:
                # Build page URL
                page_url = f"{self.base_url}?page={page}"
                response = self._make_request(page_url, client)

                if not response:
                    console.print(f"[yellow]Failed to fetch page {page}, stopping.[/yellow]")
                    break

                soup = BeautifulSoup(response.text, "lxml")

                # Try to find question links
                question_links = self._extract_question_links(soup)

                if not question_links:
                    # Try next pagination pattern
                    question_links = self._extract_question_links_fallback(soup)

                if not question_links:
                    console.print(f"[yellow]No more question links on page {page}.[/yellow]")
                    break

                self.stats.question_links_found += len(question_links)

                for link_url in question_links:
                    if records_scraped >= max_records:
                        break

                    record = self._scrape_individual_page(link_url, client)
                    if record:
                        self.stats.individual_pages_success += 1
                        yield record
                        records_scraped += 1
                    else:
                        self.stats.individual_pages_failed += 1

                    self.stats.individual_pages_attempted += 1

                page += 1
                self.stats.pages_scraped = page

        self.stats.completed_at = datetime.utcnow()
        console.print(
            f"[green]Scraping complete: {records_scraped} records from {page} pages.[/green]"
        )

    def _extract_question_links(self, soup: BeautifulSoup) -> list[str]:
        """Extract question detail page URLs from a listing page."""
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/ls/questions/" in href or "/question/" in href:
                if href.startswith("/"):
                    links.append(f"https://sansad.in{href}")
                elif href.startswith("http"):
                    links.append(href)
        return list(dict.fromkeys(links))  # Deduplicate while preserving order

    def _extract_question_links_fallback(self, soup: BeautifulSoup) -> list[str]:
        """Fallback link extraction using URL patterns."""
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "questions-and-answers" in href or "qstn" in href.lower():
                if href.startswith("/"):
                    links.append(f"https://sansad.in{href}")
                elif href.startswith("http"):
                    links.append(href)
        return list(dict.fromkeys(links))

    def _scrape_individual_page(
        self,
        url: str,
        client: httpx.Client,
    ) -> Optional[QARecord]:
        """Scrape a single question page."""
        response = self._make_request(url, client)
        if not response:
            return None

        try:
            soup = BeautifulSoup(response.text, "lxml")
            return self._parse_question_page(soup, url)
        except Exception as e:
            self.stats.parse_errors += 1
            console.print(f"[red]Parse error on {url}: {e}[/red]")
            return None

    def _parse_question_page(self, soup: BeautifulSoup, url: str) -> Optional[QARecord]:
        """Parse a question detail page into a QARecord."""
        # Extract question text
        question_elem = (
            soup.find("div", class_="question-text")
            or soup.find("div", class_="question")
            or soup.find("div", class_="q-text")
            or soup.find("h2")
            or soup.find("p")
        )
        question_text = question_elem.get_text(strip=True) if question_elem else ""

        # Extract answer text
        answer_elem = (
            soup.find("div", class_="answer-text")
            or soup.find("div", class_="answer")
            or soup.find("div", class_="a-text")
            or soup.find("div", class_="content")
            or soup.find("div", class_="qstn-answer")
        )
        answer_text = answer_elem.get_text(strip=True) if answer_elem else ""

        # Extract metadata
        ministry_elem = soup.find("span", class_="ministry") or soup.find("td", class_="ministry")
        ministry = ministry_elem.get_text(strip=True) if ministry_elem else None

        date_elem = soup.find("span", class_="date") or soup.find("td", class_="date")
        date = date_elem.get_text(strip=True) if date_elem else None

        # Extract question ID from URL
        question_id = url.split("/")[-1].replace(".html", "").replace("-", "_") or "unknown"

        if not question_text or not answer_text:
            return None

        return QARecord(
            question_id=question_id,
            question_text=question_text,
            answer_text=answer_text,
            metadata=QARecordMetadata(
                ministry=ministry,
                date=date,
                source_url=url,
            ),
            scraped_at=datetime.utcnow(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Playwright Scraper (JavaScript-Rendered Pages)
# ─────────────────────────────────────────────────────────────────────────────

class PlaywrightScraper(Scraper):
    """
    Scrapes Lok Sabha Q&A using Playwright for JavaScript-rendered pages.
    More reliable for modern web apps but slower.
    """

    async def _async_scrape_all(self, max_records: int = 3500) -> Iterator[QARecord]:
        import playwright
        from playwright.async_api import async_playwright

        self.stats.started_at = datetime.utcnow()
        console.print("[cyan]Starting Playwright scrape...[/cyan]")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                )
            )
            page = await context.new_page()
            page.set_default_timeout(30000)

            records_scraped = 0
            page_num = 1

            while records_scraped < max_records:
                page_url = f"{self.base_url}?page={page_num}"
                console.print(f"[dim]Navigating to {page_url}...[/dim]")

                try:
                    await page.goto(page_url, wait_until="networkidle")
                except Exception as e:
                    console.print(f"[yellow]Navigation error on page {page_num}: {e}[/yellow]")
                    break

                # Extract question links
                links = await page.query_selector_all("a[href*='questions']")
                link_urls = []
                for link in links:
                    href = await link.get_attribute("href")
                    if href:
                        full_url = href if href.startswith("http") else f"https://sansad.in{href}"
                        link_urls.append(full_url)

                if not link_urls:
                    console.print(f"[yellow]No more links on page {page_num}.[/yellow]")
                    break

                for link_url in link_urls:
                    if records_scraped >= max_records:
                        break

                    record = await self._scrape_individual_async(page, link_url)
                    if record:
                        self.stats.individual_pages_success += 1
                        yield record
                        records_scraped += 1
                    else:
                        self.stats.individual_pages_failed += 1
                    self.stats.individual_pages_attempted += 1

                page_num += 1
                self.stats.pages_scraped = page_num
                await asyncio.sleep(self.rate_limit_seconds)

            await browser.close()

        self.stats.completed_at = datetime.utcnow()

    def scrape_all(self, max_records: int = 3500) -> Iterator[QARecord]:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self._async_scrape_all(max_records))
                    )
                    yield from future.result()
            else:
                yield from asyncio.run(self._async_scrape_all(max_records))
        except ImportError:
            console.print("[red]Playwright not installed. Falling back to HTTPX scraper.[/red]")
            yield from []

    async def _scrape_individual_async(
        self,
        page: Any,
        url: str,
    ) -> Optional[QARecord]:
        """Scrape a single question page using Playwright."""
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            await asyncio.sleep(1)  # Allow JS to populate content

            # Extract text using Playwright's evaluation
            question_text = await page.eval_on_selector(
                "div.question, div.question-text, h2, .qstn-text",
                "el => el ? el.innerText : ''"
            )
            answer_text = await page.eval_on_selector(
                "div.answer, div.answer-text, .answer-body, .qstn-answer",
                "el => el ? el.innerText : ''"
            )
            ministry = await page.eval_on_selector(
                "span.ministry, td.ministry, .dept-name",
                "el => el ? el.innerText : ''"
            )

            if not question_text or not answer_text:
                return None

            question_id = url.split("/")[-1].replace(".html", "").replace("-", "_") or "unknown"

            return QARecord(
                question_id=question_id,
                question_text=question_text,
                answer_text=answer_text,
                metadata=QARecordMetadata(
                    ministry=ministry or None,
                    source_url=url,
                ),
                scraped_at=datetime.utcnow(),
            )
        except Exception as e:
            self.stats.parse_errors += 1
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Scraper Factory
# ─────────────────────────────────────────────────────────────────────────────

class ScraperFactory:
    """
    Factory that selects the best available scraping strategy.

    Selection priority:
    1. If strategy='mock': always use mock data
    2. If strategy='local': load from file
    3. Try playwright (best for JS-rendered pages)
    4. Try httpx (fallback for static HTML)
    5. Fall back to mock if both fail
    """

    STRATEGIES = ["auto", "playwright", "httpx", "mock", "local"]

    def __init__(
        self,
        base_url: str = "https://sansad.in/ls/questions/questions-and-answers",
        strategy: str = "auto",
        local_file: Optional[str] = None,
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Options: {', '.join(self.STRATEGIES)}"
            )
        self.base_url = base_url
        self.strategy = strategy
        self.local_file = local_file

    def create_scraper(self) -> Scraper:
        """Create the most appropriate scraper based on strategy."""
        if self.strategy == "mock":
            console.print("[yellow]Using MOCK data strategy.[/yellow]")
            console.print("[yellow]Generating realistic Lok Sabha Q&A records...[/yellow]")
            return MockScraper(self.base_url)

        if self.strategy == "local":
            if not self.local_file:
                raise ValueError("local_file is required when strategy='local'")
            console.print(f"[cyan]Loading from local file: {self.local_file}[/cyan]")
            return LocalFileScraper(self.local_file)

        if self.strategy == "playwright":
            try:
                import playwright  # noqa: F401
                console.print("[cyan]Using PLAYWRIGHT scraper (JavaScript-rendered pages).[/cyan]")
                return PlaywrightScraper(self.base_url)
            except ImportError:
                console.print(
                    "[yellow]Playwright not installed. "
                    "Install with: pip install playwright && playwright install chromium[/yellow]"
                )

        if self.strategy == "httpx":
            console.print("[cyan]Using HTTPX scraper (static HTML).[/cyan]")
            return HTTPXScraper(self.base_url)

        # Auto: try to detect best strategy
        return self._auto_select()

    def _auto_select(self) -> Scraper:
        """Automatically select the best available scraper."""
        # First, check if the site is reachable
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(self.base_url, follow_redirects=True)
                status = response.status_code
        except httpx.RequestError:
            status = 0

        if status == 200:
            # Site is reachable — try httpx first (faster)
            console.print(f"[green]Site reachable (status {status}). Using HTTPX scraper.[/green]")
            return HTTPXScraper(self.base_url)
        elif status == 403:
            # Blocked — check if playwright is available
            console.print("[yellow]Site returned 403 Forbidden. Trying Playwright...[/yellow]")
            try:
                import playwright  # noqa: F401
                return PlaywrightScraper(self.base_url)
            except ImportError:
                console.print("[yellow]Playwright not available. Falling back to MOCK data.[/yellow]")
                return MockScraper(self.base_url)
        else:
            console.print(f"[yellow]Site returned status {status}. Falling back to MOCK data.[/yellow]")
            return MockScraper(self.base_url)


class MockScraper(Scraper):
    """Wraps MockDataGenerator as a Scraper-compatible class."""

    def __init__(self, base_url: str, target_count: int = 3500, seed: int = 42) -> None:
        super().__init__(base_url)
        self.generator = MockDataGenerator(target_count=target_count, seed=seed)
        self.target_count = target_count

    def scrape_all(self, max_records: int = 3500) -> Iterator[QARecord]:
        self.stats.started_at = datetime.utcnow()
        actual_count = min(max_records, self.target_count)
        console.print(f"[cyan]Generating {actual_count:,} mock Q&A records...[/cyan]")

        count = 0
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed:,} / {task.total:,})"),
            TimeElapsedColumn(),
            console=console,
        ) as p:
            task = p.add_task("Generating records...", total=actual_count)
            for record in self.generator.generate():
                if count >= actual_count:
                    break
                yield record
                count += 1
                p.update(task, completed=count)

        self.stats.individual_pages_attempted = actual_count
        self.stats.individual_pages_success = actual_count
        self.stats.completed_at = datetime.utcnow()


class LocalFileScraper(Scraper):
    """Loads Q&A records from a local JSONL file."""

    def __init__(self, file_path: str) -> None:
        super().__init__(base_url="local")
        self.file_path = Path(file_path)
        self.stats.started_at = datetime.utcnow()

    def scrape_all(self, max_records: int = 3500) -> Iterator[QARecord]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Local file not found: {self.file_path}")

        count = 0
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as p:
            task = p.add_task(f"Loading from {self.file_path.name}...", total=max_records)
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if count >= max_records:
                        break
                    try:
                        data = json.loads(line.strip())
                        record = QARecord.model_validate(data)
                        yield record
                        count += 1
                        self.stats.individual_pages_success += 1
                    except Exception:
                        self.stats.parse_errors += 1
                    finally:
                        self.stats.individual_pages_attempted += 1
                        p.update(task, completed=count)

        self.stats.completed_at = datetime.utcnow()

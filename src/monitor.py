#!/usr/bin/env python3
"""
Real-time terminal monitoring UI for face analysis app
"""

import sqlite3
import time
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

try:
    from tqdm import tqdm
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError as e:
    print(f"Error: Required packages not installed. Please install with: pip install tqdm colorama")
    sys.exit(1)

class Monitor:
    def __init__(self, db_path: str, update_interval: float = 0.5, max_workers: int = 8):
        self.db_path = db_path
        self.update_interval = update_interval
        self.max_workers = max_workers
        self.last_stats = {}
        self.running = True

        # Initialize database connection
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

        # Print header once
        self.print_header()

    def print_header(self):
        """Print the monitoring header"""
        print(Fore.CYAN + "=" * 70)
        print(Fore.CYAN + "  Face Analysis App - 진행 상황")
        print(Fore.CYAN + "=" * 70)

    def get_total_stats(self):
        """Get total statistics from database"""
        try:
            cursor = self.conn.cursor()

            # Get total files
            cursor.execute("SELECT COUNT(*) as total FROM files")
            total_files = cursor.fetchone()['total']

            # Get processed files
            cursor.execute("SELECT COUNT(*) as processed FROM files WHERE status != 'pending'")
            processed_files = cursor.fetchone()['processed']

            # Get classified files (auto_classification = True)
            cursor.execute("SELECT COUNT(*) as auto_classified FROM files WHERE auto_classification = 1")
            auto_classified = cursor.fetchone()['auto_classified']

            # Get review pending files
            cursor.execute("SELECT COUNT(*) as review_pending FROM files WHERE status = 'review_pending'")
            review_pending = cursor.fetchone()['review_pending']

            # Get error files
            cursor.execute("SELECT COUNT(*) as error_files FROM files WHERE status = 'error'")
            error_files = cursor.fetchone()['error_files']

            # Get current batch info
            cursor.execute("""
                SELECT batch_number, file_path
                FROM files
                WHERE status IN ('processing', 'queued')
                ORDER BY rowid
                LIMIT 1
            """)
            current_batch = cursor.fetchone()

            # Get top 5 person classifications
            cursor.execute("""
                SELECT person_id, COUNT(*) as file_count
                FROM files
                WHERE person_id IS NOT NULL AND person_id != 'unknown'
                GROUP BY person_id
                ORDER BY file_count DESC
                LIMIT 5
            """)
            top_persons = cursor.fetchall()

            return {
                'total_files': total_files,
                'processed_files': processed_files,
                'auto_classified': auto_classified,
                'review_pending': review_pending,
                'error_files': error_files,
                'current_batch': current_batch,
                'top_persons': top_persons,
                'current_threshold': 0.75  # Configurable threshold
            }
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return None

    def format_progress_bar(self, current: int, total: int, width: int = 40):
        """Format progress bar with blocks"""
        if total == 0:
            return "[" + "░" * width + "]"

        filled = int((current / total) * width)
        empty = width - filled
        filled_blocks = "█" * filled
        empty_blocks = "░" * empty

        return f"[{filled_blocks}{empty_blocks}]"

    def format_estimated_time(self, current: int, total: int, start_time: float):
        """Calculate estimated remaining time"""
        if current == 0 or total == 0:
            return "계산 중..."

        elapsed = time.time() - start_time
        rate = current / elapsed if elapsed > 0 else 0
        remaining_files = total - current

        if rate == 0:
            return "계산 중..."

        remaining_seconds = remaining_files / rate

        # Format as HH:MM
        hours, remainder = divmod(remaining_seconds, 3600)
        minutes, _ = divmod(remainder, 60)

        if hours > 0:
            return f"{int(hours)}시간 {int(minutes)}분"
        else:
            return f"{int(minutes)}분"

    def print_stats(self, stats: dict, elapsed_time: float):
        """Print formatted statistics"""
        if not stats:
            return

        print("\n" + Fore.YELLOW + "-" * 70)
        print(Fore.CYAN + "실시간 통계")
        print(Fore.YELLOW + "-" * 70)

        # Overall progress
        progress_bar = self.format_progress_bar(
            stats['processed_files'],
            stats['total_files']
        )
        percentage = (stats['processed_files'] / stats['total_files'] * 100) if stats['total_files'] > 0 else 0

        print(Fore.WHITE + f"전체 진행률: {progress_bar} {percentage:.1f}% ({stats['processed_files']} / {stats['total_files']} 파일)")
        print(Fore.WHITE + f"예상 남은 시간: {self.format_estimated_time(stats['processed_files'], stats['total_files'], elapsed_time)}")

        # Current batch info
        if stats['current_batch']:
            print(Fore.WHITE + f"현재 배치: Batch #{stats['current_batch']['batch_number']}")
            print(Fore.WHITE + f"현재 파일: {stats['current_batch']['file_path']}")
        else:
            print(Fore.WHITE + "현재 배치: 처리 중인 파일 없음")

        print(Fore.YELLOW + "-" * 70)

        # Statistics details
        print(Fore.GREEN + f"✓ 처리 완료:     {stats['processed_files']} 파일")

        auto_classified_pct = (stats['auto_classified'] / stats['processed_files'] * 100) if stats['processed_files'] > 0 else 0
        print(Fore.GREEN + f"✓ 자동 분류:      {stats['auto_classified']} 파일 ({auto_classified_pct:.1f}%)")

        review_pct = (stats['review_pending'] / stats['processed_files'] * 100) if stats['processed_files'] > 0 else 0
        print(Fore.YELLOW + f"⚠ 검토 대기:      {stats['review_pending']} 파일 ({review_pct:.1f}%)")

        error_pct = (stats['error_files'] / stats['total_files'] * 100) if stats['total_files'] > 0 else 0
        print(Fore.RED + f"❌ 에러 발생:        {stats['error_files']} 파일 ({error_pct:.1f}%)")

        # Top persons
        if stats['top_persons']:
            print(Fore.WHITE + "\n인물별 분류 현황 (Top 5):")
            person_strs = []
            for person in stats['top_persons']:
                person_strs.append(f"#{person['person_id']}: {person['file_count']} 파일")
            print(" | ".join(person_strs))

        print(Fore.YELLOW + "-" * 70)
        print(Fore.WHITE + f"현재 Threshold: {stats['current_threshold']}")
        print(Fore.WHITE + f"병렬 워커: {self.max_workers}개")

    def start(self):
        """Start the monitoring display"""
        start_time = time.time()

        try:
            while self.running:
                # Get current stats
                stats = self.get_total_stats()
                if stats:
                    self.print_stats(stats, start_time)

                # Wait for update interval
                time.sleep(self.update_interval)

                # Clear screen except header
                os.system('clear')
                self.print_header()

        except KeyboardInterrupt:
            self.running = False
            print(Fore.CYAN + "\nMonitoring stopped by user.")
        except Exception as e:
            print(Fore.RED + f"Error: {e}")
        finally:
            self.conn.close()

def main():
    parser = argparse.ArgumentParser(description="Face Analysis App Monitor")
    parser.add_argument('--db', '-d', default='face_analysis.db',
                       help='SQLite database path (default: face_analysis.db)')
    parser.add_argument('--interval', '-i', type=float, default=0.5,
                       help='Update interval in seconds (default: 0.5)')
    parser.add_argument('--workers', '-w', type=int, default=8,
                       help='Number of parallel workers (default: 8)')

    args = parser.parse_args()

    # Check if database exists
    if not Path(args.db).exists():
        print(Fore.RED + f"Error: Database not found at {args.db}")
        print(Fore.YELLOW + "Please run the face analysis app first to create the database.")
        sys.exit(1)

    # Start monitor
    monitor = Monitor(args.db, args.interval, args.workers)
    monitor.start()

if __name__ == "__main__":
    main()
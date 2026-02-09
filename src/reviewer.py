"""
GUI Review Module

Tkinter-based interface for manual review of low-confidence face detections.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import shutil

from database import DatabaseManager
from detector import FaceDetector

logger = logging.getLogger(__name__)


class PersonSearchList(ttk.Frame):
    """Searchable list of persons."""

    def __init__(self, parent, num_persons: int = 35, **kwargs):
        super().__init__(parent, **kwargs)

        self.num_persons = num_persons
        self.persons = {}
        self.selected_person_id = None

        # Search entry
        search_frame = ttk.Frame(self)
        search_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self._on_search)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Person list
        list_frame = ttk.Frame(self)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                  selectmode=tk.SINGLE, height=15)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)

        # Bind selection event
        self.listbox.bind('<<ListboxSelect>>', self._on_select)

        # Initialize person list
        self._initialize_persons()

    def _initialize_persons(self):
        """Initialize person list."""
        self.listbox.delete(0, tk.END)

        for i in range(1, self.num_persons + 1):
            person_name = f"person_{i:02d}"
            self.persons[person_name] = i
            self.listbox.insert(tk.END, person_name)

    def _on_search(self, *args):
        """Handle search input."""
        search_term = self.search_var.get().lower()

        self.listbox.delete(0, tk.END)

        for person_name, person_id in self.persons.items():
            if search_term in person_name.lower():
                self.listbox.insert(tk.END, person_name)

    def _on_select(self, event):
        """Handle person selection."""
        selection = self.listbox.curselection()

        if selection:
            index = selection[0]
            person_name = self.listbox.get(index)
            self.selected_person_id = self.persons.get(person_name)
            logger.debug(f"Selected person: {person_name} (ID: {self.selected_person_id})")

    def get_selected_person_id(self) -> Optional[int]:
        """Get selected person ID."""
        return self.selected_person_id

    def clear_selection(self):
        """Clear selection."""
        self.listbox.selection_clear(0, tk.END)
        self.selected_person_id = None


class FaceReviewerGUI:
    """
    GUI for reviewing face detections.
    """

    def __init__(self, database: DatabaseManager,
                 detector: FaceDetector,
                 output_dir: str,
                 review_queue_dir: str = "./review_queue/"):
        """
        Initialize the reviewer GUI.

        Args:
            database: DatabaseManager instance
            detector: FaceDetector instance
            output_dir: Output directory for classified files
            review_queue_dir: Directory for files needing review
        """
        self.database = database
        self.detector = detector
        self.output_dir = Path(output_dir)
        self.review_queue_dir = Path(review_queue_dir)
        self.review_queue_dir.mkdir(parents=True, exist_ok=True)

        # Load detections needing review
        self.review_detections = self.database.get_detections_needing_review()
        self.current_index = 0

        # Setup GUI
        self.root = tk.Tk()
        self.root.title("Face Classification Reviewer")
        self.root.geometry("1200x800")

        self._setup_gui()

        logger.info(f"Reviewer GUI initialized with {len(self.review_detections)} items to review")

    def _setup_gui(self):
        """Setup GUI layout."""
        # Main layout
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top: Image display
        image_frame = ttk.LabelFrame(main_frame, text="Image Display")
        image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(image_frame, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bottom: Controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Person list on the left
        self.person_list = PersonSearchList(control_frame, num_persons=35)
        self.person_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Buttons on the right
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        ttk.Button(button_frame, text="Assign & Next",
                  command=self.assign_and_next).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Skip",
                  command=self.skip).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Previous",
                  command=self.previous).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Exit",
                  command=self.exit).pack(fill=tk.X, pady=2)

        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Canvas click binding
        self.canvas.bind('<Button-1>', self._on_canvas_click)

        # Load first item
        if self.review_detections:
            self._load_current_item()

    def _load_current_item(self):
        """Load current review item."""
        if self.current_index >= len(self.review_detections):
            self.status_var.set("No more items to review")
            messagebox.showinfo("Review Complete", "All items have been reviewed!")
            return

        detection = self.review_detections[self.current_index]
        image_path = detection['original_path']

        # Load and display image
        try:
            image = cv2.imread(image_path)

            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                self.skip()
                return

            # Resize to fit canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                scale = min(canvas_width / image.shape[1],
                           canvas_height / image.shape[0])
                new_width = int(image.shape[1] * scale)
                new_height = int(image.shape[0] * scale)
                image = cv2.resize(image, (new_width, new_height))

            # Convert to RGB and create PhotoImage
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            self.photo = ImageTk.PhotoImage(image_pil)

            # Display on canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            # Draw bounding box if available
            bbox = (detection['bbox_x1'], detection['bbox_y1'],
                   detection['bbox_x2'], detection['bbox_y2'])

            if bbox != (0, 0, 0, 0):
                self.canvas.create_rectangle(
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    outline='yellow', width=3
                )

                # Show current assignment
                person_id = detection.get('person_id')
                if person_id:
                    self.canvas.create_text(
                        bbox[0], bbox[1] - 10,
                        text=f"person_{person_id:02d}",
                        fill='yellow', font=('Arial', 12, 'bold')
                    )

            # Update status
            self.status_var.set(f"Item {self.current_index + 1}/{len(self.review_detections)}: "
                              f"{Path(image_path).name}")

            # Clear person selection
            self.person_list.clear_selection()

        except Exception as e:
            logger.error(f"Error loading image: {e}")
            self.skip()

    def _on_canvas_click(self, event):
        """Handle canvas click for manual bbox drawing or person assignment."""
        if self.current_index >= len(self.review_detections):
            return

        # Get selected person
        person_id = self.person_list.get_selected_person_id()

        if person_id is None:
            messagebox.showwarning("No Selection", "Please select a person from the list first")
            return

        # Assign person to current detection
        detection = self.review_detections[self.current_index]

        # Update database
        import sqlite3
        try:
            conn = sqlite3.connect(self.database.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE detections
                SET person_id = ?, needs_review = 0
                WHERE detection_id = ?
            """, (person_id, detection['detection_id']))

            conn.commit()
            conn.close()

            logger.info(f"Assigned person_{person_id:02d} to detection {detection['detection_id']}")

            # Copy file to output
            self._copy_to_output(detection['original_path'], person_id)

            # Visual feedback
            self.canvas.create_text(
                event.x, event.y,
                text=f"✓ person_{person_id:02d}",
                fill='green', font=('Arial', 14, 'bold')
            )

            # Auto-advance after short delay
            self.root.after(500, self.assign_and_next)

        except Exception as e:
            logger.error(f"Error assigning person: {e}")
            messagebox.showerror("Error", f"Failed to assign person: {e}")

    def _copy_to_output(self, file_path: str, person_id: int):
        """Copy file to output directory."""
        try:
            source = Path(file_path)
            person_dir = self.output_dir / f"person_{person_id:02d}"
            person_dir.mkdir(parents=True, exist_ok=True)

            target = person_dir / source.name

            if not target.exists():
                shutil.copy2(source, target)
                logger.debug(f"Copied {source.name} to person_{person_id:02d}")

        except Exception as e:
            logger.error(f"Failed to copy file: {e}")

    def assign_and_next(self):
        """Assign selected person and move to next item."""
        self.current_index += 1
        self._load_current_item()

    def skip(self):
        """Skip current item."""
        self.current_index += 1
        self._load_current_item()

    def previous(self):
        """Go to previous item."""
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current_item()

    def exit(self):
        """Exit the reviewer."""
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            self.root.destroy()

    def run(self):
        """Run the GUI main loop."""
        if self.review_detections:
            self.root.mainloop()
        else:
            messagebox.showinfo("No Items", "No items need review")
            self.root.destroy()


def launch_reviewer(database_path: str, output_dir: str,
                    model_name: str = "buffalo_l"):
    """
    Launch the GUI reviewer.

    Args:
        database_path: Path to database
        output_dir: Output directory
        model_name: InsightFace model name
    """
    # Initialize components
    database = DatabaseManager(database_path)
    detector = FaceDetector(model_name=model_name)

    # Create and run GUI
    reviewer = FaceReviewerGUI(
        database=database,
        detector=detector,
        output_dir=output_dir
    )
    reviewer.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    launch_reviewer(
        database_path="./data/metadata.db",
        output_dir="./classified_output/"
    )

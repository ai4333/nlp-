#!/usr/bin/env python3
"""
Automated High-Resolution Screen Recording Script for EDI
Uses Playwright with real Chrome to record a real, interactive video of the entire app,
then uses ffmpeg to export a high-definition MP4.
"""
import os, sys, time, glob, subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT_DIR = Path("/Users/nikhilsridhara/nlp/recordings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MP4 = Path("/Users/nikhilsridhara/nlp/demo_walkthrough.mp4")

def record_walkthrough():
    print("Launching Chrome in 1920x1080 to record true video...")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(
            viewport={"width": 1600, "height": 950},
            record_video_dir=str(OUT_DIR),
            record_video_size={"width": 1600, "height": 950}
        )
        page = context.new_page()

        print("1. Navigating to http://localhost:8520...")
        page.goto("http://localhost:8520", wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(4000)

        # 1. Workbench Overview
        print("2. Exploring Workbench & Asking question...")
        # Check if suggestion button exists
        sugg_buttons = page.locator("button:has-text('Explain this whole drawing')")
        if sugg_buttons.count() > 0:
            print("   Clicking 'Explain this whole drawing'...")
            sugg_buttons.first.click()
            page.wait_for_timeout(8000)
        else:
            other_btn = page.locator(".sugg button")
            if other_btn.count() > 0:
                print("   Clicking suggestion chip...")
                other_btn.first.click()
                page.wait_for_timeout(8000)

        # Scroll down chat panel
        page.mouse.wheel(0, 400)
        page.wait_for_timeout(2000)

        # Open evidence expander if present
        expander = page.locator("text='evidence & query'")
        if expander.count() > 0:
            print("   Opening Evidence & Query drawer...")
            expander.first.click()
            page.wait_for_timeout(3000)
            page.mouse.wheel(0, 300)
            page.wait_for_timeout(2000)

        # 2. Field Mode
        print("3. Switching to Field Mode...")
        field_tab = page.locator("button[role='tab']:has-text('Field mode')")
        if field_tab.count() > 0:
            field_tab.click()
            page.wait_for_timeout(3000)

            # Try clicking a suggestion in Field Mode
            field_sugg = page.locator("button:has-text('Which valves isolate')")
            if field_sugg.count() > 0:
                print("   Clicking isolation query in Field mode...")
                field_sugg.first.click()
                page.wait_for_timeout(5000)
                # Scroll down slowly to show the isolation plan and card
                for _ in range(5):
                    page.mouse.wheel(0, 250)
                    page.wait_for_timeout(1000)
                page.mouse.wheel(0, -1000)
                page.wait_for_timeout(1500)

        # 3. Knowledge Graph
        print("4. Switching to Knowledge Graph Tab...")
        kg_tab = page.locator("button[role='tab']:has-text('Knowledge graph')")
        if kg_tab.count() > 0:
            kg_tab.click()
            page.wait_for_timeout(4000)
            page.mouse.wheel(0, 200)
            page.wait_for_timeout(2000)

        # 4. Edit Graph
        print("5. Switching to Edit Graph Tab...")
        edit_tab = page.locator("button[role='tab']:has-text('Edit graph')")
        if edit_tab.count() > 0:
            edit_tab.click()
            page.wait_for_timeout(4000)

        # 5. Rig Builder
        print("6. Switching to Rig Builder Tab...")
        rig_tab = page.locator("button[role='tab']:has-text('Rig builder')")
        if rig_tab.count() > 0:
            rig_tab.click()
            page.wait_for_timeout(3000)
            # Click a package button
            pkg_btn = page.locator("button:has-text('Two-stage')")
            if pkg_btn.count() == 0:
                pkg_btn = page.locator("button:has-text('Separation')")
            if pkg_btn.count() > 0:
                print("   Selecting starter rig package...")
                pkg_btn.first.click()
                page.wait_for_timeout(4000)
                page.mouse.wheel(0, 400)
                page.wait_for_timeout(3000)

        # 6. Review Queue
        print("7. Switching to Review Queue Tab...")
        rev_tab = page.locator("button[role='tab']:has-text('Review queue')")
        if rev_tab.count() > 0:
            rev_tab.click()
            page.wait_for_timeout(3000)
            page.mouse.wheel(0, 300)
            page.wait_for_timeout(2000)

        # 7. Evaluation
        print("8. Switching to Evaluation Tab...")
        eval_tab = page.locator("button[role='tab']:has-text('Evaluation')")
        if eval_tab.count() > 0:
            eval_tab.click()
            page.wait_for_timeout(3000)
            page.mouse.wheel(0, 400)
            page.wait_for_timeout(3000)

        # Return to Workbench
        print("9. Returning to Workbench...")
        wb_tab = page.locator("button[role='tab']:has-text('Workbench')")
        if wb_tab.count() > 0:
            wb_tab.click()
            page.wait_for_timeout(3000)

        print("Finishing recording session...")
        page.close()
        context.close()
        browser.close()

    # Find the recorded video file
    recorded_videos = sorted(OUT_DIR.glob("*.webm"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not recorded_videos:
        print("Error: No recorded webm video found!")
        return False

    latest_webm = recorded_videos[0]
    print(f"Recorded raw video: {latest_webm} ({latest_webm.stat().st_size} bytes)")

    # Convert to MP4 using ffmpeg for universal compatibility
    print(f"Converting to high-compatibility MP4: {FINAL_MP4}...")
    cmd = [
        "/opt/homebrew/bin/ffmpeg", "-y",
        "-i", str(latest_webm),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "22",
        str(FINAL_MP4)
    ]
    subprocess.run(cmd, check=True)
    print(f"SUCCESS! Final MP4 created at {FINAL_MP4} ({FINAL_MP4.stat().st_size} bytes)")
    return True

if __name__ == "__main__":
    success = record_walkthrough()
    if not success:
        sys.exit(1)

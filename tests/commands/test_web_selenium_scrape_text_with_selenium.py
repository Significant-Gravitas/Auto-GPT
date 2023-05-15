# Date: 2023-5-13
# Author: Generated by GoCodeo.


import unittest
from unittest.mock import MagicMock, patch
from selenium.webdriver.remote.webdriver import WebDriver
from autogpt.commands.web_selenium import scrape_text_with_selenium, CFG
patch_func1 = "autogpt.commands.web_selenium.BeautifulSoup"

class TestScrapeTextWithSelenium(unittest.TestCase):

    @patch("selenium.webdriver.remote.webdriver")
    @patch("webdriver_manager.chrome.ChromeDriverManager")
    def test_positive_scrape_text_with_selenium_chrome(self, mock_chrome_driver_manager, mock_webdriver):
        url = "https://example.com"
        mock_driver = MagicMock()
        mock_webdriver.Chrome.return_value = mock_driver
        mock_chrome_driver_manager.return_value.install.return_value = "/path/to/chromedriver"
        CFG.selenium_web_browser = "chrome"

        with patch(patch_func1) as mock_soup:
            mock_soup_instance = MagicMock()
            mock_soup.return_value = mock_soup_instance
            mock_soup_instance.get_text.return_value = "Sample Text"

            driver, text = scrape_text_with_selenium(url)

        self.assertEqual(driver, mock_driver)
        self.assertEqual(text, "Sample Text")

    @patch("selenium.webdriver.Firefox")
    @patch("webdriver_manager.firefox.GeckoDriverManager")
    def test_positive_scrape_text_with_selenium_firefox(self, mock_gecko_driver_manager, mock_webdriver):
        url = "https://example.com"
        mock_driver = MagicMock()
        mock_webdriver.Firefox.return_value = mock_driver
        mock_gecko_driver_manager.return_value.install.return_value = "/path/to/geckodriver"
        CFG.selenium_web_browser = "firefox"

        with patch(patch_func1) as mock_soup:
            mock_soup_instance = MagicMock()
            mock_soup.return_value = mock_soup_instance
            mock_soup_instance.get_text.return_value = "Sample Text"

            driver, text = scrape_text_with_selenium(url)

        self.assertEqual(driver, mock_driver)
        self.assertEqual(text, "Sample Text")

    @patch("selenium.webdriver.Safari")
    def test_positive_scrape_text_with_selenium_safari(self, mock_webdriver):
        url = "https://example.com"
        mock_driver = MagicMock()
        mock_webdriver.return_value = mock_driver
        CFG.selenium_web_browser = "safari"

        with patch(patch_func1) as mock_soup:
            mock_soup_instance = MagicMock()
            mock_soup.return_value = mock_soup_instance
            mock_soup_instance.get_text.return_value = "Sample Text"

            driver, text = scrape_text_with_selenium(url)

        self.assertEqual(driver, mock_driver)
        self.assertEqual(text, "Sample Text")

    def test_negative_scrape_text_with_selenium_invalid_browser(self):
        url = "https://example.com"
        CFG.selenium_web_browser = "invalid_browser"

        with self.assertRaises(KeyError):
            driver, text = scrape_text_with_selenium(url)

    @patch("selenium.webdriver.Chrome")
    @patch("webdriver_manager.chrome.ChromeDriverManager")
    def test_scrape_text_with_selenium_empty_content(self, mock_chrome_driver_manager, mock_chrome_driver):
        mock_chrome_driver_manager.return_value.install.return_value = "/path/to/chrome/driver"
        mock_chrome_driver.return_value.execute_script.return_value = "<html><body></body></html>"

        url = "https://example.com"
        driver, text = scrape_text_with_selenium(url)

        self.assertIsInstance(driver, WebDriver)
        self.assertEqual(text, "")

if __name__ == "__main__":
    unittest.main()
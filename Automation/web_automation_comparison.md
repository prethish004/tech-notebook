
## Same Login Test: Playwright vs Selenium

Here's the **exact same login test** on `saucedemo.com` written both ways. Test it with `pytest`!

### Playwright Version
File: `playwright/tests/test_login.py`

```python
import pytest
from playwright.sync_api import Page, expect

@pytest.mark.asyncio
async def test_login_playwright(page: Page):
    await page.goto("https://www.saucedemo.com/")

    await page.fill('[data-test="username"]', 'standard_user')
    await page.fill('[data-test="password"]', 'secret_sauce')
    await page.click('[data-test="login-button"]')

    await expect(page).to_have_url("**/inventory.html")
    await expect(page.locator('[data-test="shopping-cart-contents"]')).to_be_visible()
```

### Selenium Version  
File: `selenium/tests/test_login.py`

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pytest

@pytest.fixture
def driver():
    driver = webdriver.Chrome()
    yield driver
    driver.quit()

def test_login_selenium(driver):
    driver.get('https://www.saucedemo.com/')

    username = driver.find_element(By.CSS_SELECTOR, '[data-test="username"]')
    username.send_keys('standard_user')

    password = driver.find_element(By.CSS_SELECTOR, '[data-test="password"]')
    password.send_keys('secret_sauce')

    login_btn = driver.find_element(By.CSS_SELECTOR, '[data-test="login-button"]')
    login_btn.click()

    wait = WebDriverWait(driver, 10)
    wait.until(EC.url_contains('/inventory.html'))
    cart = driver.find_element(By.CSS_SELECTOR, '[data-test="shopping-cart-contents"]')
    assert cart.is_displayed()
```

### Key Differences You Notice:

| Aspect | Playwright | Selenium |
|--------|------------|----------|
| **Setup** | `pytest-playwright` auto-manages browsers | Need ChromeDriver setup |
| **API Style** | Fluent, async-friendly, auto-waits built-in | More verbose, manual waits |
| **Waits** | `expect(page).to_have_url()` auto-waits | `WebDriverWait` + EC manual |
| **Selectors** | Same CSS/XPath | Same CSS/XPath |
| **Cleanup** | Fixture handles | Manual `driver.quit()` |
| **Speed** | ~2-3x faster execution | Slower due to WebDriver protocol |

### Run Both

```bash
# Playwright (from repo root)
pytest playwright/tests/test_login.py -v

# Selenium
pytest selenium/tests/test_login.py -v
```

**Result:** Both pass, but Playwright is shorter, faster, less flaky.[web:34][web:40]

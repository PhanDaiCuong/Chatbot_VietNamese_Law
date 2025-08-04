from playwright.sync_api import sync_playwright
import time
import json

def crawl_table_with_pagination(url, max_pages=3):
    all_data = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # Mở để dễ debug - hiển thị 
        page = browser.new_page()
        page.goto(url)
        page.wait_for_timeout(2000)
        items = [ 'Số kí hiệu' , 'Ngày ban hành' , 'Cơ quan ban hành' , 'Trích yếu']
        for page_num in range(1, max_pages + 1):

            # Chờ bảng load
            page.wait_for_selector("#grid_ThuTuc")

            # Crawl bảng hiện tại
            rows = page.locator("#grid_ThuTuc tbody tr")
            for i in range(rows.count()):
                row = rows.nth(i)
                cells = row.locator("td")
                row_data = {}

                for j in range(1, cells.count() - 1 ):  
                    cell_text = cells.nth(j).inner_text().strip()
                    row_data[items[j-1]] = cell_text 

                all_data.append(row_data)

            # Chuyển sang trang tiếp theo (nếu chưa là trang cuối)
            if page_num < max_pages:
                page.click(f'li.paginate_button >> text="{page_num + 1}"')
                page.wait_for_timeout(2000)  
                
        browser.close()
    return all_data

if __name__ == "__main__":
    url = "https://qlvb.moj.gov.vn/cddh/Pages/chidaodieuhanh.aspx"
    data = crawl_table_with_pagination(url, max_pages=3)

    with open("data_example.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Đã lưu xong dữ liệu phân trang.")

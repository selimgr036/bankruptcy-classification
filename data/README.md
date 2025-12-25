# Bankruptcy Prediction Dataset

A novel dataset for bankruptcy prediction related to American public companies listed on the New York Stock Exchange and NASDAQ is provided. The dataset comprises accounting data from 8,262 distinct companies recorded during the period spanning from 1999 to 2018.

## Bankruptcy Definition

According to the Security Exchange Commission (SEC), a company in the American market is deemed bankrupt under two circumstances. Firstly, if the firm's management files for Chapter 11 of the Bankruptcy Code, indicating an intention to "reorganize" its business. In this case, the company's management continues to oversee day-to-day operations, but significant business decisions necessitate approval from a bankruptcy court. Secondly, if the firm's management files for Chapter 7 of the Bankruptcy Code, indicating a complete cessation of operations and the company going out of business entirely.

In this dataset, the fiscal year prior to the filing of bankruptcy under either Chapter 11 or Chapter 7 is labeled as "Bankruptcy" (1) for the subsequent year. Conversely, if the company does not experience these bankruptcy events, it is considered to be operating normally (0). The dataset is complete, without any missing values, synthetic entries, or imputed added values.

## Dataset Structure

The resulting dataset comprises a total of 78,682 observations of firm-year combinations. To facilitate model training and evaluation, the dataset is divided into three subsets based on time periods:

- **Training set**: Data from 1999 to 2011
- **Validation set**: Data from 2012 to 2014
- **Test set**: Data from 2015 to 2018

The test set serves as a means to assess the predictive capability of models in real-world scenarios involving unseen cases.

## Variables

| Variable Name | Description |
|---------------|-------------|
| X1 | Current assets - All the assets of a company that are expected to be sold or used as a result of standard business operations over the next year |
| X2 | Cost of goods sold - The total amount a company paid as a cost directly related to the sale of products |
| X3 | Depreciation and amortization - Depreciation refers to the loss of value of a tangible fixed asset over time (such as property, machinery, buildings, and plant). Amortization refers to the loss of value of intangible assets over time. |
| X4 | EBITDA - Earnings before interest, taxes, depreciation, and amortization. It is a measure of a company's overall financial performance, serving as an alternative to net income. |
| X5 | Inventory - The accounting of items and raw materials that a company either uses in production or sells. |
| X6 | Net Income - The overall profitability of a company after all expenses and costs have been deducted from total revenue. |
| X7 | Total Receivables - The balance of money due to a firm for goods or services delivered or used but not yet paid for by customers. |
| X8 | Market value - The price of an asset in a marketplace. In this dataset, it refers to the market capitalization since companies are publicly traded in the stock market. |
| X9 | Net sales - The sum of a company's gross sales minus its returns, allowances, and discounts. |
| X10 | Total assets - All the assets, or items of value, a business owns. |
| X11 | Total Long-term debt - A company's loans and other liabilities that will not become due within one year of the balance sheet date. |
| X12 | EBIT - Earnings before interest and taxes. |
| X13 | Gross Profit - The profit a business makes after subtracting all the costs that are related to manufacturing and selling its products or services. |
| X14 | Total Current Liabilities - The sum of accounts payable, accrued liabilities, and taxes such as Bonds payable at the end of the year, salaries, and commissions remaining. |
| X15 | Retained Earnings - The amount of profit a company has left over after paying all its direct costs, indirect costs, income taxes, and its dividends to shareholders. |
| X16 | Total Revenue - The amount of income that a business has made from all sales before subtracting expenses. It may include interest and dividends from investments. |
| X17 | Total Liabilities - The combined debts and obligations that the company owes to outside parties. |
| X18 | Total Operating Expenses - The expenses a business incurs through its normal business operations. |


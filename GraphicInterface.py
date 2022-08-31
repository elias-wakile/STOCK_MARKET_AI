import datetime
import tkinter
from tkcalendar import Calendar
from tkinter import ttk, PhotoImage, messagebox
import datetime
import yfinance as yf
import AI_NeuralNetork_Trader



def create_entry_window():
    def display_msg():
        format = '%m/%d/%y'
        sd = datetime.datetime.strptime(start_date.get_date(), format)
        ed = datetime.datetime.strptime(end_date.get_date(), format)
        delte_date = ed - sd
        if delte_date.days < 7:
            messagebox.showerror("Date error", "Please select two dates with at least a difference of at least 7 days.")
            return
        stock_names = stock_entry.get().replace(" ", "").upper().split(",")
        if len(stock_names) == 1 and stock_names[0] == "":
            messagebox.showerror("No Stock name error", "Please enter the stock symbols separated by comma"
                                                     " (For Example: AAPL, GOOGL)")
            return
        tickers = []
        for t in stock_names:
            tickers.append(yf.Ticker(t))
            if tickers[-1].info['regularMarketPrice'] is None:
                messagebox.showerror("Stock name error", "One of the stock symbol is invalid, Please enter the stock "
                                                         "symbols separated by comma (For Example: AAPL, GOOGL)")

                return
        window.destroy()
        AI_NeuralNetork_Trader.main_def(sd, ed, stock_names, tickers)

    window = tkinter.Tk()
    window.title('The Wall Street Orca')
    window.geometry('600x400+50+50')
    window.resizable(False, False)
    window.minsize(600, 600)
    icon = PhotoImage(file="icon.png")
    window.iconphoto(False, icon)
    logo = PhotoImage(file="Logo.png")
    tkinter.Label(window, image=logo).grid(row=0, column=0, columnspan=2)
    start_date = Calendar(window, selectmode='day',
                          maxdate=datetime.date.today(),
                          mindate=datetime.date.today() - datetime.timedelta(days=31),
                          state='normal')
    start_label = tkinter.Label(window, text='Begin Investment On:')
    start_label.grid(row=1, column=0, padx=20, pady=10)
    start_date.grid(row=2, column=0, padx=35, pady=10)
    end_date = Calendar(window, selectmode='day',
                          maxdate=datetime.date.today(),
                          mindate=datetime.date.today() - datetime.timedelta(days=31),
                          state='normal')
    end_label = tkinter.Label(window, text='End Investment On:')
    end_label.grid(row=1, column=1, padx=20, pady=10)
    end_date.grid(row=2, column=1, padx=35, pady=10)
    stock_label = tkinter.Label(window, text='Please enter the stock symbols separated by comma (For Example:'
                                             ' AAPL, GOOGL)')
    stock_entry = tkinter.Entry(window, {'width':50})
    stock_label.grid(row=3, column=0, columnspan=2, pady=10, padx=10)
    stock_entry.grid(row=4, column=0, columnspan=2, pady=0)
    ttk.Button(window, text='Submit', command=display_msg).place(relx=0.825, rely=0.9)
    window.mainloop()



if __name__ == '__main__':
    create_entry_window()